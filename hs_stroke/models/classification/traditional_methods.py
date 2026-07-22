"""Traditional MI classification methods."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from mne.decoding import CSP
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


TARGET_FS = 128
SPD_EPS = 1e-5
FBCSP_BANDS = [(10, 13), (13, 16), (16, 19), (19, 22), (22, 25)]
TWFB_BANDS = [
    (8, 12),
    (8, 20),
    (8, 30),
    (12, 20),
    (15, 20),
    (15, 30),
    (20, 30),
    (8, 15),
]


def notch_filter(data: np.ndarray, fs: int, f0: float) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=30.0, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def bandpass_filter(
    data: np.ndarray,
    fs: int,
    low: float,
    high: float,
) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def apply_filter_pipeline(
    data: np.ndarray,
    fs: int,
    low: float,
    high: float,
) -> np.ndarray:
    filtered = notch_filter(data, fs=fs, f0=50.0)
    return np.asarray(
        bandpass_filter(filtered, fs=fs, low=low, high=high),
        dtype=np.float64,
    )


def compute_covariances(data: np.ndarray) -> np.ndarray:
    covariance = np.matmul(data, np.transpose(data, (0, 2, 1)))
    covariance = 0.5 * (covariance + np.transpose(covariance, (0, 2, 1)))
    identity = np.eye(covariance.shape[1], dtype=np.float64)[None, :, :]
    traces = np.trace(covariance, axis1=1, axis2=2)
    scales = np.where(traces > 0.0, traces / covariance.shape[1], 1.0)
    covariance = covariance + (SPD_EPS * scales)[:, None, None] * identity
    minimum_eigenvalues = np.linalg.eigvalsh(covariance)[:, 0]
    invalid = minimum_eigenvalues <= 0.0
    if np.any(invalid):
        correction = -minimum_eigenvalues[invalid] + SPD_EPS * scales[invalid]
        covariance[invalid] += correction[:, None, None] * identity[0]
    return covariance


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1score": float(f1score),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def fit_predict_tslda(
    covariance_train: np.ndarray,
    y_train: np.ndarray,
    covariance_eval: np.ndarray,
) -> np.ndarray:
    model = Pipeline(
        [
            ("tangent_space", TangentSpace(metric="riemann")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    model.fit(covariance_train, y_train)
    return model.predict(covariance_eval)


def fit_predict_fgmdm(
    covariance_train: np.ndarray,
    y_train: np.ndarray,
    covariance_eval: np.ndarray,
) -> np.ndarray:
    model = FgMDM(metric="riemann", tsupdate=False, n_jobs=1)
    model.fit(covariance_train, y_train)
    return model.predict(covariance_eval)


def fit_fbcsp(
    train_x: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], CSP]]]:
    transformers = []
    features = []
    for band in FBCSP_BANDS:
        band_data = apply_filter_pipeline(
            train_x,
            fs=TARGET_FS,
            low=float(band[0]),
            high=float(band[1]),
        )
        csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
        features.append(np.asarray(csp.fit_transform(band_data, y_train)))
        transformers.append((band, csp))
    return np.concatenate(features, axis=1), transformers


def transform_fbcsp(
    data: np.ndarray,
    transformers: Sequence[Tuple[Tuple[int, int], CSP]],
) -> np.ndarray:
    features = []
    for band, csp in transformers:
        band_data = apply_filter_pipeline(
            data,
            fs=TARGET_FS,
            low=float(band[0]),
            high=float(band[1]),
        )
        features.append(np.asarray(csp.transform(band_data)))
    return np.concatenate(features, axis=1)


def subset_to_numpy(subset) -> Tuple[np.ndarray, np.ndarray]:
    signals = []
    labels = []
    for index in range(len(subset)):
        signal, label = subset[index]
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()
        signals.append(np.asarray(signal, dtype=np.float64))
        labels.append(int(label))
    return np.stack(signals), np.asarray(labels, dtype=np.int64)


def run_fbcsp_svm(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    train_features, transformers = fit_fbcsp(train_x, y_train)
    test_features = transform_fbcsp(test_x, transformers)
    model = LinearSVC(max_iter=10000)
    model.fit(train_features, y_train)
    metrics = evaluate_metrics(y_test, model.predict(test_features))
    metrics["selected_backend"] = "linear_svc"
    metrics["selected_band"] = ",".join(
        f"{low}-{high}" for low, high in FBCSP_BANDS
    )
    return metrics


def run_tslda_dgfmdrm(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    train_filtered = apply_filter_pipeline(train_x, TARGET_FS, 4.0, 30.0)
    test_filtered = apply_filter_pipeline(test_x, TARGET_FS, 4.0, 30.0)
    covariance_train = compute_covariances(train_filtered)
    covariance_test = compute_covariances(test_filtered)
    tslda_accuracy = np.mean(
        fit_predict_tslda(covariance_train, y_train, covariance_train) == y_train
    )
    fgmdm_accuracy = np.mean(
        fit_predict_fgmdm(covariance_train, y_train, covariance_train) == y_train
    )
    if tslda_accuracy >= fgmdm_accuracy:
        prediction = fit_predict_tslda(covariance_train, y_train, covariance_test)
        backend = "tslda"
    else:
        prediction = fit_predict_fgmdm(covariance_train, y_train, covariance_test)
        backend = "fgmdm"
    metrics = evaluate_metrics(y_test, prediction)
    metrics["selected_backend"] = backend
    metrics["selected_band"] = "4-30"
    return metrics


def run_twfb_dgfmdrm(
    train_x: np.ndarray,
    y_train: np.ndarray,
    test_x: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    best_band = max(
        TWFB_BANDS,
        key=lambda band: np.mean(
            fit_predict_fgmdm(
                compute_covariances(
                    apply_filter_pipeline(train_x, TARGET_FS, *map(float, band))
                ),
                y_train,
                compute_covariances(
                    apply_filter_pipeline(train_x, TARGET_FS, *map(float, band))
                ),
            )
            == y_train
        ),
    )
    covariance_train = compute_covariances(
        apply_filter_pipeline(train_x, TARGET_FS, *map(float, best_band))
    )
    covariance_test = compute_covariances(
        apply_filter_pipeline(test_x, TARGET_FS, *map(float, best_band))
    )
    prediction = fit_predict_fgmdm(covariance_train, y_train, covariance_test)
    metrics = evaluate_metrics(y_test, prediction)
    metrics["selected_backend"] = "fgmdm"
    metrics["selected_band"] = f"{best_band[0]}-{best_band[1]}"
    return metrics


ML_RUNNERS = {
    "fbcsp_svm": run_fbcsp_svm,
    "tslda_dgfmdrm": run_tslda_dgfmdrm,
    "twfb_dgfmdrm": run_twfb_dgfmdrm,
}

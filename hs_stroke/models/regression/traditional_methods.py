"""Traditional FMA regression methods."""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, filtfilt, iirnotch


FIXED_SCALE = 66.0
SOURCE_FS = 256
TARGET_FS = 128
WINDOW_SECONDS = 1
WINDOW_SAMPLES_SOURCE = SOURCE_FS * WINDOW_SECONDS
WINDOW_SAMPLES_TARGET = TARGET_FS * WINDOW_SECONDS
SPD_EPS = 1e-5
DEFAULT_METHODS = ["tslda_dgfmdrm", "fbcsp_svm", "twfb_dgfmdrm"]
LINEAR_REG_LAMBDA = 1e-3
DEFAULT_CHANNEL_LIST = [
    "FC3",
    "FC4",
    "C5",
    "C3",
    "C1",
    "CZ",
    "C2",
    "C4",
    "C6",
    "CP3",
    "CP4",
]
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


def notch_filter(data: np.ndarray, fs: int, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def bandpass_filter(data: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return filtfilt(b, a, data, axis=-1)


def apply_filter_pipeline(data: np.ndarray, fs: int, low: float, high: float) -> np.ndarray:
    filtered = notch_filter(data, fs=fs, f0=50.0)
    filtered = bandpass_filter(filtered, fs=fs, low=low, high=high)
    return np.asarray(filtered, dtype=np.float64)


def compute_covariances(data: np.ndarray) -> np.ndarray:
    cov = np.matmul(data, np.transpose(data, (0, 2, 1)))
    cov = 0.5 * (cov + np.transpose(cov, (0, 2, 1)))
    eye = np.eye(cov.shape[1], dtype=np.float64)[None, :, :]
    traces = np.trace(cov, axis1=1, axis2=2)
    scales = np.where(traces > 0.0, traces / cov.shape[1], 1.0)
    cov = cov + (SPD_EPS * scales)[:, None, None] * eye
    min_eigs = np.linalg.eigvalsh(cov)[:, 0]
    bad = min_eigs <= 0.0
    if np.any(bad):
        cov[bad] = cov[bad] + ((-min_eigs[bad] + SPD_EPS * scales[bad])[:, None, None] * eye[0])
    return cov


def compute_raw_covariances(data: np.ndarray) -> np.ndarray:
    cov = np.matmul(data, np.transpose(data, (0, 2, 1)))
    return 0.5 * (cov + np.transpose(cov, (0, 2, 1)))


def flatten_features(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = np.mean(np.abs(y_pred - y_true)) * FIXED_SCALE
    mse = np.mean((y_pred - y_true) ** 2) * (FIXED_SCALE ** 2)
    return {
        "mae": float(mae),
        "mse": float(mse),
        "true_fma": float(np.mean(y_true) * FIXED_SCALE),
        "predicted_fma": float(np.mean(y_pred) * FIXED_SCALE),
    }


def fit_linear_regression_and_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    lambda_reg: float = LINEAR_REG_LAMBDA,
) -> Tuple[np.ndarray, int]:
    X_train_aug = np.concatenate(
        [X_train, np.ones((X_train.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    X_eval_aug = np.concatenate(
        [X_eval, np.ones((X_eval.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    xtx = X_train_aug.T @ X_train_aug
    reg = lambda_reg * np.eye(xtx.shape[0], dtype=np.float64)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(xtx + reg, X_train_aug.T @ y_train)
    pred = X_eval_aug @ beta
    return pred, int(X_train_aug.shape[1])


def compute_train_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0.0:
        return float("-inf")
    return float(1.0 - (np.sum((y_pred - y_true) ** 2) / denom))


def run_loso_tslda_dgfmdrm(
    split,
    _: Sequence[float],
    clip_predictions: bool = False,
) -> Dict[str, object]:
    full_train_filt = apply_filter_pipeline(split.full_train_X, fs=TARGET_FS, low=4.0, high=30.0)
    test_filt = apply_filter_pipeline(split.test_X, fs=TARGET_FS, low=4.0, high=30.0)
    cov_full_train = compute_covariances(full_train_filt)
    cov_test = compute_covariances(test_filt)

    ts = TangentSpace(metric="riemann")
    X_train_ts = ts.fit_transform(cov_full_train)
    train_pred_ts, ts_dim = fit_linear_regression_and_predict(
        X_train_ts, split.full_train_y, X_train_ts, lambda_reg=LINEAR_REG_LAMBDA
    )
    ts_r2 = compute_train_r2(split.full_train_y, train_pred_ts)

    X_train_cov = flatten_features(compute_raw_covariances(full_train_filt))
    train_pred_cov, cov_dim = fit_linear_regression_and_predict(
        X_train_cov, split.full_train_y, X_train_cov, lambda_reg=LINEAR_REG_LAMBDA
    )
    cov_r2 = compute_train_r2(split.full_train_y, train_pred_cov)

    if ts_r2 >= cov_r2:
        ts_full = TangentSpace(metric="riemann")
        X_full_ts = ts_full.fit_transform(cov_full_train)
        X_test_ts = ts_full.transform(cov_test)
        test_pred = fit_linear_regression_and_predict(
            X_full_ts, split.full_train_y, X_test_ts, lambda_reg=LINEAR_REG_LAMBDA
        )[0]
        feature_dim = ts_dim
        selected_backend = "tslda"
        selected_score = ts_r2
    else:
        X_full_cov = flatten_features(compute_raw_covariances(full_train_filt))
        X_test_cov = flatten_features(compute_raw_covariances(test_filt))
        test_pred = fit_linear_regression_and_predict(
            X_full_cov, split.full_train_y, X_test_cov, lambda_reg=LINEAR_REG_LAMBDA
        )[0]
        feature_dim = cov_dim
        selected_backend = "dgfmdrm_like_cov_ridge"
        selected_score = cov_r2

    metric_pred = np.clip(test_pred, 0.0, 1.0) if clip_predictions else test_pred
    metrics = regression_metrics(split.test_y, metric_pred)
    metrics.update({
        "params": int(feature_dim),
        "selected_backend": selected_backend,
        "selected_alpha": float(LINEAR_REG_LAMBDA),
        "selected_band": "4-30",
        "selection_score_r2": float(selected_score),
    })
    return metrics


def run_loso_fbcsp_svm(
    split,
    _: Sequence[float],
    clip_predictions: bool = False,
) -> Dict[str, object]:
    full_train_filt = apply_filter_pipeline(split.full_train_X, fs=TARGET_FS, low=8.0, high=40.0)
    test_filt = apply_filter_pipeline(split.test_X, fs=TARGET_FS, low=8.0, high=40.0)
    X_full_train = flatten_features(full_train_filt)
    X_test = flatten_features(test_filt)
    test_pred, feature_dim = fit_linear_regression_and_predict(
        X_full_train, split.full_train_y, X_test, lambda_reg=LINEAR_REG_LAMBDA
    )

    metric_pred = np.clip(test_pred, 0.0, 1.0) if clip_predictions else test_pred
    metrics = regression_metrics(split.test_y, metric_pred)
    metrics.update({
        "params": int(feature_dim),
        "selected_backend": "linear_regression",
        "selected_alpha": float(LINEAR_REG_LAMBDA),
        "selected_band": "8-40",
        "selection_score_r2": float("nan"),
    })
    return metrics


def run_loso_twfb_dgfmdrm(
    split,
    _: Sequence[float],
    clip_predictions: bool = False,
) -> Dict[str, object]:
    best_band = TWFB_BANDS[0]
    best_feature_dim = 0
    best_train_r2 = float("-inf")

    for band in TWFB_BANDS:
        train_filt = apply_filter_pipeline(
            split.full_train_X, fs=TARGET_FS, low=float(band[0]), high=float(band[1])
        )
        X_train = flatten_features(compute_raw_covariances(train_filt))
        train_pred, feature_dim = fit_linear_regression_and_predict(
            X_train, split.full_train_y, X_train, lambda_reg=LINEAR_REG_LAMBDA
        )
        train_r2 = compute_train_r2(split.full_train_y, train_pred)
        if train_r2 > best_train_r2:
            best_train_r2 = train_r2
            best_band = band
            best_feature_dim = feature_dim

    full_train_filt = apply_filter_pipeline(
        split.full_train_X, fs=TARGET_FS, low=float(best_band[0]), high=float(best_band[1])
    )
    test_filt = apply_filter_pipeline(
        split.test_X, fs=TARGET_FS, low=float(best_band[0]), high=float(best_band[1])
    )
    X_full_train = flatten_features(compute_raw_covariances(full_train_filt))
    X_test = flatten_features(compute_raw_covariances(test_filt))
    test_pred, _ = fit_linear_regression_and_predict(
        X_full_train, split.full_train_y, X_test, lambda_reg=LINEAR_REG_LAMBDA
    )

    metric_pred = np.clip(test_pred, 0.0, 1.0) if clip_predictions else test_pred
    metrics = regression_metrics(split.test_y, metric_pred)
    metrics.update({
        "params": int(best_feature_dim),
        "selected_backend": "covariance_linear_regression",
        "selected_alpha": float(LINEAR_REG_LAMBDA),
        "selected_band": f"{best_band[0]}-{best_band[1]}",
        "selection_score_r2": float(best_train_r2),
    })
    return metrics


def run_losso_tslda_dgfmdrm(split, _: Sequence[float]) -> Dict[str, object]:
    result = run_loso_tslda_dgfmdrm(split, _, clip_predictions=True)
    result["feature_dim"] = int(result.pop("params"))
    result["selected_score_train_r2"] = float(result.pop("selection_score_r2"))
    result.pop("selected_alpha", None)
    if result["selected_backend"] == "dgfmdrm_like_cov_ridge":
        result["selected_backend"] = "dgfmdrm_cov"
    return result


def run_losso_fbcsp_svm(split, _: Sequence[float]) -> Dict[str, object]:
    result = run_loso_fbcsp_svm(split, _, clip_predictions=True)
    train_filt = apply_filter_pipeline(split.full_train_X, fs=TARGET_FS, low=8.0, high=40.0)
    train_features = flatten_features(train_filt)
    train_pred = fit_linear_regression_and_predict(
        train_features, split.full_train_y, train_features
    )[0]
    result["feature_dim"] = int(result.pop("params"))
    result["selected_score_train_r2"] = float(compute_train_r2(split.full_train_y, train_pred))
    result.pop("selection_score_r2", None)
    result.pop("selected_alpha", None)
    return result


def run_losso_twfb_dgfmdrm(split, _: Sequence[float]) -> Dict[str, object]:
    result = run_loso_twfb_dgfmdrm(split, _, clip_predictions=True)
    result["feature_dim"] = int(result.pop("params"))
    result["selected_score_train_r2"] = float(result.pop("selection_score_r2"))
    result.pop("selected_alpha", None)
    if result["selected_backend"] == "covariance_linear_regression":
        result["selected_backend"] = "cov_linear_regression"
    return result


METHOD_RUNNERS_LOSO = {
    "tslda_dgfmdrm": run_loso_tslda_dgfmdrm,
    "fbcsp_svm": run_loso_fbcsp_svm,
    "twfb_dgfmdrm": run_loso_twfb_dgfmdrm,
}
METHOD_RUNNERS_LOSSO = {
    "tslda_dgfmdrm": run_losso_tslda_dgfmdrm,
    "fbcsp_svm": run_losso_fbcsp_svm,
    "twfb_dgfmdrm": run_losso_twfb_dgfmdrm,
}

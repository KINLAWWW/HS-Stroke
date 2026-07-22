"""Cross-session runner for traditional MI baselines."""

from __future__ import annotations

import gc
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from mne.decoding import CSP
from pyriemann.classification import FgMDM
from pyriemann.tangentspace import TangentSpace
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Convergence not reached")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import DERIVATIVES_ROOT, RESULTS_ROOT

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
DEFAULT_METHODS = ["tslda_dgfmdrm", "fbcsp_svm", "twfb_dgfmdrm"]
TARGET_FS = 128
SOURCE_FS = 256
WINDOW_SECONDS = 1
WINDOW_SAMPLES = TARGET_FS * WINDOW_SECONDS
SPD_EPS = 1e-5

FBCSP_BANDS = [
    (10, 13),
    (13, 16),
    (16, 19),
    (19, 22),
    (22, 25),
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


@dataclass(frozen=True)
class SessionWindows:
    subject: str
    session_id: str
    session_num: int
    X: np.ndarray  # [n_windows, n_channels, n_times]
    y: np.ndarray  # [n_windows]
    n_trials: int


@dataclass(frozen=True)
class SplitArrays:
    core_train_X: np.ndarray
    core_train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    full_train_X: np.ndarray
    full_train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    train_sessions: List[int]
    core_train_sessions: List[int]
    val_sessions: List[int]
    test_sessions: List[int]


@dataclass
class ExperimentConfig:
    methods: tuple[str, ...] = tuple(DEFAULT_METHODS)
    root_path: str | Path = DERIVATIVES_ROOT
    results_root: str | Path = RESULTS_ROOT / "mi_classification" / "cross_session"
    val_session_ratio: float = 0.2
    val_min_sessions: int = 1
    max_subjects: int = 0


def _build_config(overrides: dict) -> ExperimentConfig:
    known_fields = set(ExperimentConfig.__dataclass_fields__)
    unknown = sorted(set(overrides) - known_fields)
    if unknown:
        raise TypeError(f"Unknown experiment configuration fields: {unknown}")
    config = ExperimentConfig(**overrides)
    invalid_methods = sorted(set(config.methods) - set(METHOD_RUNNERS))
    if invalid_methods:
        raise ValueError(f"Unknown methods: {invalid_methods}")
    return config


def _log(message: str) -> None:
    print(message, flush=True)


def _metric_tiebreak(metrics: Dict[str, float]) -> Tuple[float, float, float]:
    return (metrics["accuracy_pct"], metrics["f1score"], metrics["kappa"])


def _extract_channel_names(configuration_channel: np.ndarray) -> List[str]:
    names: List[str] = []
    for ch in configuration_channel[0]:
        if ch[1].sum():
            names.append(ch[0].tolist()[0])
    return names


def _session_to_int(session_id: str) -> int:
    matched = re.findall(r"(\d+)", str(session_id))
    if not matched:
        raise ValueError(f"Invalid session id: {session_id}")
    return int(matched[-1])


def _load_session_windows(mat_path: Path) -> SessionWindows:
    mat = loadmat(mat_path, verify_compressed_data_integrity=False)
    eeg_data = mat["EEGdata"]
    labels = mat["EEGdatalabel"].reshape(-1).astype(np.int64)
    ch_names = _extract_channel_names(mat["configuration_channel"])
    if ch_names != DEFAULT_CHANNEL_LIST:
        raise ValueError(f"Channel mismatch in {mat_path}: {ch_names}")

    if eeg_data.ndim != 3:
        raise ValueError(f"Unexpected EEGdata shape in {mat_path}: {eeg_data.shape}")

    if eeg_data.shape[1] == len(DEFAULT_CHANNEL_LIST):
        trials = eeg_data
    elif eeg_data.shape[0] == len(DEFAULT_CHANNEL_LIST):
        trials = np.transpose(eeg_data, (2, 0, 1))
    else:
        raise ValueError(f"Cannot infer trial/channel axis order in {mat_path}: {eeg_data.shape}")

    if trials.shape[0] != len(labels):
        raise ValueError(
            f"Label mismatch in {mat_path}: {trials.shape[0]} trials vs {len(labels)} labels"
        )

    if trials.shape[-1] != 4 * TARGET_FS:
        trials = resample_poly(trials, up=TARGET_FS, down=SOURCE_FS, axis=-1)

    if trials.shape[-1] % WINDOW_SAMPLES != 0:
        raise ValueError(f"Trial length {trials.shape[-1]} is not divisible by {WINDOW_SAMPLES}")

    num_windows_per_trial = trials.shape[-1] // WINDOW_SAMPLES
    windows = trials.reshape(trials.shape[0], trials.shape[1], num_windows_per_trial, WINDOW_SAMPLES)
    windows = np.transpose(windows, (0, 2, 1, 3)).reshape(-1, trials.shape[1], WINDOW_SAMPLES)
    window_labels = np.repeat(labels, num_windows_per_trial)

    session_id = mat_path.parts[-3]
    subject = mat_path.parts[-4]
    return SessionWindows(
        subject=subject,
        session_id=session_id,
        session_num=_session_to_int(session_id),
        X=np.asarray(windows, dtype=np.float64),
        y=window_labels,
        n_trials=int(trials.shape[0]),
    )


def load_subject_sessions(root_path: Path, subject: str) -> Dict[int, SessionWindows]:
    subject_dir = root_path / subject
    session_files = sorted(subject_dir.glob("ses-*/eeg/*.mat"))
    if not session_files:
        raise FileNotFoundError(f"No derivative MAT files found for {subject} under {subject_dir}")

    sessions: Dict[int, SessionWindows] = {}
    for mat_path in session_files:
        session = _load_session_windows(mat_path)
        sessions[session.session_num] = session
    return sessions


def split_subject_sessions(
    session_nums: Sequence[int],
) -> Tuple[List[int], List[int]]:
    session_nums = sorted(set(int(s) for s in session_nums))
    if len(session_nums) < 2:
        raise ValueError("At least two sessions are required for cross-session evaluation.")
    test_nums = session_nums[::2]
    train_nums = session_nums[1::2]
    return train_nums, test_nums


def split_train_val_sessions(
    train_session_nums: Sequence[int],
    val_session_ratio: float,
    val_min_sessions: int,
) -> Tuple[List[int], List[int]]:
    train_session_nums = sorted(set(int(s) for s in train_session_nums))
    if len(train_session_nums) < 2:
        raise ValueError("Not enough training sessions to build a validation split.")

    val_count = int(round(len(train_session_nums) * val_session_ratio))
    val_count = max(val_min_sessions, val_count)
    val_count = min(len(train_session_nums) - 1, val_count)
    val_nums = train_session_nums[-val_count:]
    core_train_nums = train_session_nums[:-val_count]
    return core_train_nums, val_nums


def concat_sessions(
    sessions: Dict[int, SessionWindows],
    session_nums: Iterable[int],
) -> Tuple[np.ndarray, np.ndarray]:
    ordered = [sessions[int(num)] for num in sorted(session_nums)]
    X = np.concatenate([session.X for session in ordered], axis=0)
    y = np.concatenate([session.y for session in ordered], axis=0)
    return X, y


def build_split_arrays(
    sessions: Dict[int, SessionWindows],
    train_session_nums: Sequence[int],
    test_session_nums: Sequence[int],
    val_session_ratio: float,
    val_min_sessions: int,
) -> SplitArrays:
    core_train_sessions, val_sessions = split_train_val_sessions(train_session_nums, val_session_ratio, val_min_sessions)
    core_train_X, core_train_y = concat_sessions(sessions, core_train_sessions)
    val_X, val_y = concat_sessions(sessions, val_sessions)
    full_train_X, full_train_y = concat_sessions(sessions, train_session_nums)
    test_X, test_y = concat_sessions(sessions, test_session_nums)
    return SplitArrays(
        core_train_X=core_train_X,
        core_train_y=core_train_y,
        val_X=val_X,
        val_y=val_y,
        full_train_X=full_train_X,
        full_train_y=full_train_y,
        test_X=test_X,
        test_y=test_y,
        train_sessions=list(sorted(int(s) for s in train_session_nums)),
        core_train_sessions=list(sorted(int(s) for s in core_train_sessions)),
        val_sessions=list(sorted(int(s) for s in val_sessions)),
        test_sessions=list(sorted(int(s) for s in test_session_nums)),
    )


def notch_filter(data: np.ndarray, fs: int, f0: float, q: float = 30.0) -> np.ndarray:
    b, a = iirnotch(w0=f0, Q=q, fs=fs)
    return filtfilt(b, a, data, axis=-1)


def bandpass_filter(data: np.ndarray, fs: int, low: float, high: float, order: int = 4) -> np.ndarray:
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


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


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    precision, recall, f1score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy_pct": 100.0 * float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1score": float(f1score),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def fit_predict_tslda(cov_train: np.ndarray, y_train: np.ndarray, cov_test: np.ndarray) -> np.ndarray:
    model = Pipeline(
        [
            ("ts", TangentSpace(metric="riemann")),
            ("lda", LinearDiscriminantAnalysis()),
        ]
    )
    model.fit(cov_train, y_train)
    return model.predict(cov_test)


def fit_predict_fgmdm(cov_train: np.ndarray, y_train: np.ndarray, cov_test: np.ndarray) -> np.ndarray:
    model = FgMDM(metric="riemann", tsupdate=False, n_jobs=1)
    model.fit(cov_train, y_train)
    return model.predict(cov_test)


def run_tslda_dgfmdrm(split_arrays: SplitArrays) -> Dict[str, float]:
    train_X = apply_filter_pipeline(split_arrays.core_train_X, fs=TARGET_FS, low=4.0, high=30.0)
    val_X = apply_filter_pipeline(split_arrays.val_X, fs=TARGET_FS, low=4.0, high=30.0)

    cov_train = compute_covariances(train_X)
    cov_val = compute_covariances(val_X)
    train_y = split_arrays.core_train_y
    val_y = split_arrays.val_y

    candidate_metrics: Dict[str, Dict[str, float]] = {}
    for backend, predictor in {
        "tslda": fit_predict_tslda,
        "fgmdm": fit_predict_fgmdm,
    }.items():
        val_pred = predictor(cov_train, train_y, cov_val)
        candidate_metrics[backend] = evaluate_metrics(val_y, val_pred)

    best_backend = max(candidate_metrics, key=lambda name: _metric_tiebreak(candidate_metrics[name]))

    full_train_X = apply_filter_pipeline(split_arrays.full_train_X, fs=TARGET_FS, low=4.0, high=30.0)
    test_X = apply_filter_pipeline(split_arrays.test_X, fs=TARGET_FS, low=4.0, high=30.0)
    cov_full_train = compute_covariances(full_train_X)
    cov_test = compute_covariances(test_X)

    if best_backend == "tslda":
        test_pred = fit_predict_tslda(cov_full_train, split_arrays.full_train_y, cov_test)
    else:
        test_pred = fit_predict_fgmdm(cov_full_train, split_arrays.full_train_y, cov_test)

    metrics = evaluate_metrics(split_arrays.test_y, test_pred)
    metrics["selected_backend"] = best_backend
    return metrics


def fit_fbcsp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    bands: Sequence[Tuple[int, int]],
    csp_pairs: int = 2,
) -> Tuple[np.ndarray, List[Tuple[Tuple[int, int], CSP]]]:
    transformers: List[Tuple[Tuple[int, int], CSP]] = []
    features: List[np.ndarray] = []
    for band in bands:
        X_band = apply_filter_pipeline(X_train, fs=TARGET_FS, low=float(band[0]), high=float(band[1]))
        csp = CSP(n_components=2 * csp_pairs, reg=None, log=True, norm_trace=False)
        band_features = csp.fit_transform(X_band, y_train)
        features.append(np.asarray(band_features, dtype=np.float64))
        transformers.append((band, csp))
    return np.concatenate(features, axis=1), transformers


def transform_fbcsp(
    X: np.ndarray,
    transformers: Sequence[Tuple[Tuple[int, int], CSP]],
) -> np.ndarray:
    features: List[np.ndarray] = []
    for band, csp in transformers:
        X_band = apply_filter_pipeline(X, fs=TARGET_FS, low=float(band[0]), high=float(band[1]))
        band_features = csp.transform(X_band)
        features.append(np.asarray(band_features, dtype=np.float64))
    return np.concatenate(features, axis=1)


def run_fbcsp_svm(split_arrays: SplitArrays) -> Dict[str, float]:
    train_features, transformers = fit_fbcsp(split_arrays.core_train_X, split_arrays.core_train_y, FBCSP_BANDS)
    svm_val = LinearSVC(max_iter=10000)
    svm_val.fit(train_features, split_arrays.core_train_y)
    val_features = transform_fbcsp(split_arrays.val_X, transformers)
    _ = evaluate_metrics(split_arrays.val_y, svm_val.predict(val_features))

    full_train_features, full_transformers = fit_fbcsp(
        split_arrays.full_train_X, split_arrays.full_train_y, FBCSP_BANDS
    )
    test_features = transform_fbcsp(split_arrays.test_X, full_transformers)
    svm_test = LinearSVC(max_iter=10000)
    svm_test.fit(full_train_features, split_arrays.full_train_y)
    test_pred = svm_test.predict(test_features)

    metrics = evaluate_metrics(split_arrays.test_y, test_pred)
    metrics["selected_backend"] = "linear_svc"
    metrics["selected_band"] = ",".join([f"{low}-{high}" for low, high in FBCSP_BANDS])
    return metrics


def run_twfb_dgfmdrm(split_arrays: SplitArrays) -> Dict[str, float]:
    band_scores: Dict[Tuple[int, int], Dict[str, float]] = {}
    for band in TWFB_BANDS:
        train_X = apply_filter_pipeline(
            split_arrays.core_train_X, fs=TARGET_FS, low=float(band[0]), high=float(band[1])
        )
        val_X = apply_filter_pipeline(split_arrays.val_X, fs=TARGET_FS, low=float(band[0]), high=float(band[1]))
        cov_train = compute_covariances(train_X)
        cov_val = compute_covariances(val_X)
        val_pred = fit_predict_fgmdm(cov_train, split_arrays.core_train_y, cov_val)
        band_scores[band] = evaluate_metrics(split_arrays.val_y, val_pred)

    best_band = max(band_scores, key=lambda band: _metric_tiebreak(band_scores[band]))

    full_train_X = apply_filter_pipeline(
        split_arrays.full_train_X, fs=TARGET_FS, low=float(best_band[0]), high=float(best_band[1])
    )
    test_X = apply_filter_pipeline(
        split_arrays.test_X, fs=TARGET_FS, low=float(best_band[0]), high=float(best_band[1])
    )
    cov_full_train = compute_covariances(full_train_X)
    cov_test = compute_covariances(test_X)
    test_pred = fit_predict_fgmdm(cov_full_train, split_arrays.full_train_y, cov_test)

    metrics = evaluate_metrics(split_arrays.test_y, test_pred)
    metrics["selected_backend"] = "fgmdm"
    metrics["selected_band"] = f"{best_band[0]}-{best_band[1]}"
    return metrics


METHOD_RUNNERS = {
    "tslda_dgfmdrm": run_tslda_dgfmdrm,
    "fbcsp_svm": run_fbcsp_svm,
    "twfb_dgfmdrm": run_twfb_dgfmdrm,
}


def summarize_rows(rows: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    metric_cols = ["accuracy_pct", "precision", "recall", "f1score", "kappa"]

    summary_records = []
    for metric in metric_cols:
        values = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        summary_records.append(
            {
                "metric": metric.replace("_pct", ""),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    summary_df = pd.DataFrame(summary_records)

    mean_std_row: Dict[str, object] = {"subject": "MEAN±STD"}
    for col in df.columns:
        if col == "subject":
            continue
        if col in metric_cols:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            mean_std_row[col] = f"{np.mean(values):.6f} ± {np.std(values, ddof=1):.6f}"
        elif col.startswith("n_"):
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            mean_std_row[col] = f"{np.mean(values):.1f}"
        else:
            mean_std_row[col] = ""

    df_with_summary = pd.concat([df, pd.DataFrame([mean_std_row])], ignore_index=True)
    return df_with_summary, summary_df


def _discover_subjects(root_path: Path) -> List[str]:
    subjects = sorted([p.name for p in root_path.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if not subjects:
        raise FileNotFoundError(f"No subject folders found under {root_path}")
    return subjects


def run_experiment(**overrides) -> None:
    args = _build_config(overrides)

    root_path = Path(args.root_path).resolve()
    results_root = Path(args.results_root).resolve()

    subject_cache: Dict[str, Dict[int, SessionWindows]] = {}
    result_rows: Dict[Tuple[str, str], List[Dict[str, object]]] = {
        ("cross_session", method): [] for method in args.methods
    }

    subjects = _discover_subjects(root_path)
    if args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]

    start_time = time.perf_counter()
    _log(
        "[start] ML cross-session runner "
        f"subjects={len(subjects)} methods={args.methods} strategy=cross_session"
    )

    for subject_idx, subject in enumerate(subjects, start=1):
        subject_cache[subject] = load_subject_sessions(root_path, subject)
        session_nums = sorted(subject_cache[subject].keys())
        _log(f"[subject {subject_idx}/{len(subjects)}] {subject} sessions={session_nums}")

        for strategy in ["cross_session"]:
            train_session_nums, test_session_nums = split_subject_sessions(
                session_nums=session_nums,
            )
            split_arrays = build_split_arrays(
                sessions=subject_cache[subject],
                train_session_nums=train_session_nums,
                test_session_nums=test_session_nums,
                val_session_ratio=args.val_session_ratio,
                val_min_sessions=args.val_min_sessions,
            )

            _log(
                f"[split] subject={subject} strategy={strategy} "
                f"train={split_arrays.train_sessions} val={split_arrays.val_sessions} "
                f"test={split_arrays.test_sessions} "
                f"samples(train/val/test)="
                f"{len(split_arrays.full_train_y)}/{len(split_arrays.val_y)}/{len(split_arrays.test_y)}"
            )

            for method in args.methods:
                method_start = time.perf_counter()
                metrics = METHOD_RUNNERS[method](split_arrays)
                elapsed = time.perf_counter() - method_start

                row: Dict[str, object] = {
                    "subject": subject,
                    "strategy": strategy,
                    "method": method,
                    "train_sessions": "-".join(map(str, split_arrays.train_sessions)),
                    "core_train_sessions": "-".join(map(str, split_arrays.core_train_sessions)),
                    "val_sessions": "-".join(map(str, split_arrays.val_sessions)),
                    "test_sessions": "-".join(map(str, split_arrays.test_sessions)),
                    "n_core_train": int(len(split_arrays.core_train_y)),
                    "n_val": int(len(split_arrays.val_y)),
                    "n_full_train": int(len(split_arrays.full_train_y)),
                    "n_test": int(len(split_arrays.test_y)),
                    "elapsed_sec": round(elapsed, 3),
                }
                row.update(metrics)
                result_rows[(strategy, method)].append(row)

                _log(
                    f"[done] subject={subject} strategy={strategy} method={method} "
                    f"acc={metrics['accuracy_pct']:.3f} f1={metrics['f1score']:.3f} "
                    f"kappa={metrics['kappa']:.3f} extra="
                    f"{metrics.get('selected_backend', '')}{('/' + metrics['selected_band']) if 'selected_band' in metrics else ''} "
                    f"time={elapsed:.1f}s"
                )

            del split_arrays
            gc.collect()

        gc.collect()

    for (strategy, method), rows in result_rows.items():
        if not rows:
            continue

        results_dir = results_root / method
        results_dir.mkdir(parents=True, exist_ok=True)
        per_subject_df, summary_df = summarize_rows(rows)
        per_subject_path = results_dir / "subject_results.csv"
        summary_path = results_dir / "summary.csv"

        per_subject_df.to_csv(per_subject_path, index=False)
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")

        metric_lookup = {
            row["metric"]: (row["mean"], row["std"])
            for _, row in summary_df.iterrows()
        }
        _log(
            f"[write] strategy={strategy} method={method} "
            f"acc={metric_lookup['accuracy'][0]:.3f} ± {metric_lookup['accuracy'][1]:.3f} "
            f"-> {summary_path.name}"
        )

    elapsed_total = time.perf_counter() - start_time
    _log(f"[finished] total_time={elapsed_total / 60.0:.2f} min")

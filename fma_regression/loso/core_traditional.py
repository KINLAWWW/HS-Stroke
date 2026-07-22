"""LOSO FMA traditional runner."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hs_stroke.datasets.fma import FMAAssessment, load_fma_assessments
from hs_stroke.models.regression.traditional_methods import (
    DEFAULT_CHANNEL_LIST,
    DEFAULT_METHODS,
    FIXED_SCALE,
    METHOD_RUNNERS_LOSO as METHOD_RUNNERS,
    SOURCE_FS,
    WINDOW_SAMPLES_SOURCE,
    WINDOW_SAMPLES_TARGET,
    notch_filter,
)
from experiment_config import FMA_SESSIONS_TSV, FMA_SOURCE_ROOT, PARTICIPANTS_TSV, RESULTS_ROOT


@dataclass(frozen=True)
class SubjectWindows:
    subject_id: str
    X: np.ndarray  # [n_windows, n_channels, n_times]
    y: np.ndarray  # [n_windows] normalized to [0, 1]


@dataclass(frozen=True)
class FoldSplit:
    full_train_X: np.ndarray
    full_train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def _log(msg: str) -> None:
    print(msg, flush=True)


def _extract_channel_names(configuration_channel: np.ndarray) -> List[str]:
    names: List[str] = []
    for ch in configuration_channel[0]:
        if ch[1].sum():
            names.append(ch[0].tolist()[0])
    return names


def load_subject_windows(
    assessments: Sequence[FMAAssessment],
    subject_id: str,
) -> SubjectWindows:
    windows: List[np.ndarray] = []
    labels: List[float] = []

    subject_assessments = [
        item for item in assessments if item.participant_id == subject_id
    ]
    for assessment in sorted(
        subject_assessments,
        key=lambda item: item.fma_session_id,
    ):
        fma = assessment.fma_score / FIXED_SCALE
        for mat_file in assessment.mat_files:
            mat = loadmat(mat_file, verify_compressed_data_integrity=False)
            eeg_data = mat["EEGdata"]  # [C, T, N]
            ch_names = _extract_channel_names(mat["configuration_channel"])
            if ch_names != DEFAULT_CHANNEL_LIST:
                raise ValueError(f"Channel mismatch in {mat_file}: {ch_names}")

            if eeg_data.ndim != 3:
                raise ValueError(f"Unexpected EEGdata shape in {mat_file}: {eeg_data.shape}")

            num_trials = eeg_data.shape[2]
            for trial_idx in range(num_trials):
                trial_data = eeg_data[:, 9 * SOURCE_FS:13 * SOURCE_FS, trial_idx]
                trial_data = notch_filter(trial_data, fs=SOURCE_FS, f0=50.0)
                for window_idx in range(4):
                    start = window_idx * WINDOW_SAMPLES_SOURCE
                    end = start + WINDOW_SAMPLES_SOURCE
                    window = trial_data[:, start:end]
                    window = resample_poly(window, up=1, down=2, axis=1)
                    if window.shape[1] != WINDOW_SAMPLES_TARGET:
                        raise ValueError(f"Unexpected window length after resample: {window.shape}")
                    windows.append(np.asarray(window, dtype=np.float64))
                    labels.append(float(fma))

    if not windows:
        raise RuntimeError(f"No valid windows loaded for subject {subject_id}")

    X = np.stack(windows, axis=0)
    y = np.asarray(labels, dtype=np.float64)
    return SubjectWindows(subject_id=subject_id, X=X, y=y)


def discover_subjects(assessments: Sequence[FMAAssessment]) -> List[str]:
    subjects = sorted({item.participant_id for item in assessments})
    if not subjects:
        raise FileNotFoundError("No subjects found in the FMA session map")
    return subjects


def build_fold_split(
    subject_data: Dict[str, SubjectWindows],
    test_subject: str,
) -> FoldSplit:
    train_X_list: List[np.ndarray] = []
    train_y_list: List[np.ndarray] = []
    for subject_id, data in subject_data.items():
        if subject_id == test_subject:
            continue
        train_X_list.append(data.X)
        train_y_list.append(data.y)

    full_train_X = np.concatenate(train_X_list, axis=0)
    full_train_y = np.concatenate(train_y_list, axis=0)

    test_data = subject_data[test_subject]
    return FoldSplit(
        full_train_X=full_train_X,
        full_train_y=full_train_y,
        test_X=test_data.X,
        test_y=test_data.y,
    )


def append_average_row(df: pd.DataFrame) -> pd.DataFrame:
    avg_row: Dict[str, object] = {col: "" for col in df.columns}
    avg_row["subject"] = "AVERAGE"
    if "mae" in df.columns:
        avg_row["mae"] = float(df["mae"].astype(float).mean())
    if "mse" in df.columns:
        avg_row["mse"] = float(df["mse"].astype(float).mean())
    return pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)


def summarize_subject_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in ["mae", "mse"]:
        vals = df[metric].astype(float).to_numpy(dtype=float)
        rows.append({
            "metric": metric,
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0),
        })
    return pd.DataFrame(rows)


def run_experiment(
    *,
    methods=tuple(DEFAULT_METHODS),
    root_path=FMA_SOURCE_ROOT,
    participants_path=PARTICIPANTS_TSV,
    session_map_path=FMA_SESSIONS_TSV,
    results_root=RESULTS_ROOT / "fma_regression" / "loso",
    max_folds=0,
) -> None:
    unknown_methods = sorted(set(methods) - set(METHOD_RUNNERS))
    if unknown_methods:
        raise ValueError(f"Unknown traditional FMA methods: {unknown_methods}")

    root_path = Path(root_path).resolve()
    results_root = Path(results_root)

    assessments = load_fma_assessments(
        sourcedata_root=root_path,
        participants_path=participants_path,
        session_map_path=session_map_path,
    )
    subjects = discover_subjects(assessments)
    subject_data = {
        subject_id: load_subject_windows(assessments, subject_id)
        for subject_id in subjects
    }

    total_windows = sum(len(data.y) for data in subject_data.values())
    _log(
        f"[start] LOSO traditional FMA regression "
        f"subjects={len(subjects)} total_windows={total_windows} methods={methods}"
    )

    for method in methods:
        results_dir = results_root / method
        results_dir.mkdir(parents=True, exist_ok=True)
        results_path = results_dir / "predictions.csv"
        summary_path = results_dir / "summary.csv"

        subject_rows: List[Dict[str, object]] = []

        for fold_idx, test_subject in enumerate(subjects, start=1):
            if max_folds > 0 and fold_idx > max_folds:
                break

            split = build_fold_split(
                subject_data=subject_data,
                test_subject=test_subject,
            )

            _log(
                f"[fold {fold_idx:02d}/{len(subjects)}] method={method} "
                f"test_subject={test_subject} train={len(split.full_train_y)} "
                f"test={len(split.test_y)}"
            )

            fold_start = time.perf_counter()
            result = METHOD_RUNNERS[method](split, ())
            elapsed = time.perf_counter() - fold_start

            subject_row: Dict[str, object] = {
                "subject": test_subject,
                "fold": float(fold_idx),
                "params": float(result["params"]),
                "mae": float(result["mae"]),
                "mse": float(result["mse"]),
                "true_fma": float(result["true_fma"]),
                "predicted_fma": float(result["predicted_fma"]),
                "selected_backend": result["selected_backend"],
                "selected_alpha": float(result["selected_alpha"]),
                "selected_band": result["selected_band"],
                "selection_score_r2": float(result["selection_score_r2"]) if str(result["selection_score_r2"]) != "nan" else np.nan,
                "elapsed_sec": round(elapsed, 3),
            }
            subject_rows.append(subject_row)

            _log(
                f"[done] method={method} test_subject={test_subject} "
                f"mae={subject_row['mae']:.3f} mse={subject_row['mse']:.3f} "
                f"backend={subject_row['selected_backend']} "
                f"band={subject_row['selected_band']} alpha={subject_row['selected_alpha']} "
                f"train_r2={subject_row['selection_score_r2']} "
                f"time={elapsed:.1f}s"
            )

        if not subject_rows:
            raise RuntimeError(f"No folds were executed for method {method}.")

        subject_df = pd.DataFrame(subject_rows)
        summary_df = summarize_subject_rows(subject_df)
        results_with_avg = append_average_row(subject_df)

        results_with_avg.to_csv(results_path, index=False, float_format="%.6f")
        summary_df.to_csv(summary_path, index=False, float_format="%.6f")

        metric_lookup = {row["metric"]: (row["mean"], row["std"]) for _, row in summary_df.iterrows()}
        _log(
            f"[write] method={method} mae={metric_lookup['mae'][0]:.3f} ± {metric_lookup['mae'][1]:.3f} "
            f"mse={metric_lookup['mse'][0]:.3f} ± {metric_lookup['mse'][1]:.3f}"
        )

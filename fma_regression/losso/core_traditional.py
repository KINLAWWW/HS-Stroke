"""LOSSO FMA traditional runner."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
    METHOD_RUNNERS_LOSSO as METHOD_RUNNERS,
    SOURCE_FS,
    WINDOW_SAMPLES_SOURCE,
    notch_filter,
)
from experiment_config import FMA_SESSIONS_TSV, FMA_SOURCE_ROOT, PARTICIPANTS_TSV, RESULTS_ROOT


@dataclass(frozen=True)
class SessionWindows:
    subject_id: str
    session_id: str
    X: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class FoldSplit:
    full_train_X: np.ndarray
    full_train_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    test_subject: str
    test_session: str
    same_subject_train_sessions: List[str]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _extract_channel_names(configuration_channel: np.ndarray) -> List[str]:
    names: List[str] = []
    for ch in configuration_channel[0]:
        if ch[1].sum():
            names.append(ch[0].tolist()[0])
    return names


def load_session_windows(assessment: FMAAssessment) -> SessionWindows:
    windows: List[np.ndarray] = []
    labels: List[float] = []
    fma = assessment.fma_score / FIXED_SCALE

    for mat_file in assessment.mat_files:
        mat = loadmat(mat_file, verify_compressed_data_integrity=False)
        eeg_data = mat["EEGdata"]
        ch_names = _extract_channel_names(mat["configuration_channel"])
        if ch_names != DEFAULT_CHANNEL_LIST:
            raise ValueError(f"Channel mismatch in {mat_file}: {ch_names}")

        num_trials = eeg_data.shape[2]
        for trial_idx in range(num_trials):
            trial_data = eeg_data[:, 9 * SOURCE_FS:13 * SOURCE_FS, trial_idx]
            trial_data = notch_filter(trial_data, fs=SOURCE_FS, f0=50.0)
            for window_idx in range(4):
                start = window_idx * WINDOW_SAMPLES_SOURCE
                end = start + WINDOW_SAMPLES_SOURCE
                window = trial_data[:, start:end]
                window = resample_poly(window, up=1, down=2, axis=1)
                windows.append(np.asarray(window, dtype=np.float64))
                labels.append(float(fma))

    if not windows:
        raise RuntimeError(
            f"No valid windows loaded for session {assessment.fma_session_id}"
        )

    return SessionWindows(
        subject_id=assessment.participant_id,
        session_id=assessment.fma_session_id,
        X=np.stack(windows, axis=0),
        y=np.asarray(labels, dtype=np.float64),
    )


def discover_sessions(
    assessments: Sequence[FMAAssessment],
) -> List[Tuple[str, str]]:
    discovered = sorted(
        (item.participant_id, item.fma_session_id)
        for item in assessments
    )
    if not discovered:
        raise FileNotFoundError("No FMA assessments found in the session map")
    return discovered


def load_all_sessions(
    assessments: Sequence[FMAAssessment],
) -> Dict[Tuple[str, str], SessionWindows]:
    cache: Dict[Tuple[str, str], SessionWindows] = {}
    for assessment in assessments:
        key = (assessment.participant_id, assessment.fma_session_id)
        if key in cache:
            raise ValueError(f"Duplicate FMA assessment key: {key}")
        cache[key] = load_session_windows(assessment)
    return cache


def build_fold_split(
    session_data: Dict[Tuple[str, str], SessionWindows],
    test_subject: str,
    test_session: str,
) -> FoldSplit:
    train_X_list: List[np.ndarray] = []
    train_y_list: List[np.ndarray] = []
    same_subject_train_sessions: List[str] = []

    for (subject_id, session_id), session in session_data.items():
        if subject_id == test_subject and session_id == test_session:
            continue
        train_X_list.append(session.X)
        train_y_list.append(session.y)
        if subject_id == test_subject:
            same_subject_train_sessions.append(session_id)

    test_session_data = session_data[(test_subject, test_session)]
    return FoldSplit(
        full_train_X=np.concatenate(train_X_list, axis=0),
        full_train_y=np.concatenate(train_y_list, axis=0),
        test_X=test_session_data.X,
        test_y=test_session_data.y,
        test_subject=test_subject,
        test_session=test_session,
        same_subject_train_sessions=sorted(same_subject_train_sessions),
    )


def summarize(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        values = df[metric].astype(float).to_numpy()
        rows.append({
            "metric": metric,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        })
    return pd.DataFrame(rows)


def run_experiment(
    *,
    methods=tuple(DEFAULT_METHODS),
    root_path=FMA_SOURCE_ROOT,
    participants_path=PARTICIPANTS_TSV,
    session_map_path=FMA_SESSIONS_TSV,
    results_root=RESULTS_ROOT / "fma_regression" / "losso",
    max_subject_sessions=0,
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
    session_data = load_all_sessions(assessments)

    metrics = ["mae", "mse"]
    subject_sessions = sorted(session_data.keys())
    _log(f"[start] loaded {len(subject_sessions)} subject-session folds from {root_path}")

    for method in methods:
        results_dir = results_root / method
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv = results_dir / "predictions.csv"
        summary_csv = results_dir / "summary.csv"

        rows: List[Dict[str, object]] = []

        for fold_idx, (test_subject, test_session) in enumerate(subject_sessions, start=1):
            if max_subject_sessions > 0 and fold_idx > max_subject_sessions:
                break

            fold_start = time.perf_counter()
            split = build_fold_split(session_data, test_subject, test_session)
            method_result = METHOD_RUNNERS[method](split, ())
            elapsed = time.perf_counter() - fold_start

            row = {
                "subject_session": f"{test_subject}/{test_session}",
                "test_subject": test_subject,
                "test_session": test_session,
                "fold": fold_idx,
                "same_subject_train_sessions": "|".join(split.same_subject_train_sessions),
                "elapsed_sec": round(elapsed, 3),
            }
            row.update(method_result)
            rows.append(row)

            _log(
                f"[done] method={method} fold={fold_idx} subject={test_subject} session={test_session} "
                f"mae={method_result['mae']:.3f} mse={method_result['mse']:.3f} time={elapsed:.1f}s"
            )

        if not rows:
            raise RuntimeError(f"No results produced for method {method}.")

        result_df = pd.DataFrame(rows)
        result_df.to_csv(results_csv, index=False, float_format="%.6f")
        summarize(result_df, metrics).to_csv(summary_csv, index=False, float_format="%.6f")
        _log(f"[write] method={method} -> {results_csv.name}")

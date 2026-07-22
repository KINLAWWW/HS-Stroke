"""MI cross-trial traditional runner."""

from __future__ import annotations

import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from torcheeg import transforms

from experiment_config import CACHE_ROOT, DERIVATIVES_ROOT, MI_CROSS_TRIAL_CACHE
from hs_stroke.datasets.derivatives import DerivativesDataset
from hs_stroke.splits import CrossTrialSplit
from hs_stroke.models.classification.traditional_methods import (
    ML_RUNNERS,
    subset_to_numpy,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CACHE = CACHE_ROOT / "splits" / "mi_cross_trial"

N_SPLITS = 5
RANDOM_SEED = 42
OVERLAP = 0.5
NUMERIC_THREADS = 2
METRICS = ["accuracy", "recall", "precision", "f1score", "kappa"]
SPLIT_PROTOCOL = "cross_trial"
SUPPORTED_METHODS = {"tslda_dgfmdrm", "fbcsp_svm", "twfb_dgfmdrm"}


class TeeStream:
    def __init__(self, console: TextIO, log_file: TextIO) -> None:
        self.console = console
        self.log_file = log_file

    def write(self, message: str) -> None:
        self.console.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self) -> None:
        self.console.flush()
        self.log_file.flush()


def result_dir(method: str) -> Path:
    return WORKSPACE_ROOT / "results" / "mi_classification" / "cross_trial" / method


def log_dir(method: str) -> Path:
    return WORKSPACE_ROOT / "logs" / "mi_classification" / "cross_trial" / method


def build_dataset() -> DerivativesDataset:
    if not DERIVATIVES_ROOT.is_dir():
        raise FileNotFoundError(f"Data root not found: {DERIVATIVES_ROOT}")
    return DerivativesDataset(
        root_path=str(DERIVATIVES_ROOT),
        io_path=str(MI_CROSS_TRIAL_CACHE),
        overlap=OVERLAP,
        num_worker=0,
        label_transform=transforms.Compose([transforms.Select("label")]),
    )


def write_results(method: str, fold_rows: list[dict]) -> None:
    output_dir = result_dir(method)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = output_dir / "fold_results.csv"
    subject_csv = output_dir / "subject_results.csv"

    fold_df = pd.DataFrame(fold_rows).sort_values(["subject", "session", "fold"])
    fold_df.to_csv(fold_csv, index=False, float_format="%.6f")

    subject_df = (
        fold_df.groupby("subject", as_index=False)[METRICS]
        .mean()
        .sort_values("subject")
    )
    fold_counts = fold_df.groupby("subject").size().rename("completed_folds")
    subject_df.insert(
        1,
        "completed_folds",
        subject_df["subject"].map(fold_counts).astype(int),
    )
    mean_row = {
        "subject": "MEAN",
        "completed_folds": int(subject_df["completed_folds"].sum()),
    }
    for metric in METRICS:
        mean_row[metric] = float(subject_df[metric].mean())
    pd.concat([subject_df, pd.DataFrame([mean_row])], ignore_index=True).to_csv(
        subject_csv,
        index=False,
        float_format="%.6f",
    )


def load_existing_rows(method: str) -> list[dict]:
    fold_csv = result_dir(method) / "fold_results.csv"
    if not fold_csv.is_file():
        return []
    fold_df = pd.read_csv(fold_csv)
    if fold_df[["subject", "session", "fold"]].astype(str).duplicated().any():
        raise RuntimeError(f"Resume CSV contains duplicate folds: {fold_csv}")
    return fold_df.to_dict(orient="records")


def run_experiment(method: str) -> None:
    method = method.lower()
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported traditional method: {method}")

    np.random.seed(RANDOM_SEED)
    dataset = build_dataset()
    splitter = CrossTrialSplit(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
        split_path=str(SPLIT_CACHE),
    )
    fold_rows = load_existing_rows(method)
    completed = {
        (str(row["subject"]), str(row["session"]), int(row["fold"]))
        for row in fold_rows
    }

    print(
        f"[start] method={method} samples={len(dataset)} protocol={SPLIT_PROTOCOL}",
        flush=True,
    )
    for train_set, val_set, subject, session, fold_id in splitter.split(dataset):
        fold = int(fold_id) + 1
        key = (str(subject), str(session), fold)
        if key in completed:
            continue

        train_x, train_y = subset_to_numpy(train_set)
        val_x, val_y = subset_to_numpy(val_set)
        with threadpool_limits(limits=NUMERIC_THREADS):
            metrics = ML_RUNNERS[method](train_x, train_y, val_x, val_y)

        row = {
            "subject": str(subject),
            "session": str(session),
            "fold": fold,
            "train_samples": len(train_x),
            "val_samples": len(val_x),
            "evaluation_role": "validation",
            "selection_protocol": "fixed_method_no_epoch_selection",
            "split_protocol": SPLIT_PROTOCOL,
            "selected_backend": metrics.pop("selected_backend", ""),
            "selected_band": metrics.pop("selected_band", ""),
            **{metric: metrics[metric] for metric in METRICS},
        }
        fold_rows.append(row)
        write_results(method, fold_rows)
        print(
            f"[result] subject={subject} session={session} fold={fold} "
            f"accuracy={row['accuracy']:.6f} f1={row['f1score']:.6f} "
            f"kappa={row['kappa']:.6f}",
            flush=True,
        )
        del train_x, train_y, val_x, val_y
        gc.collect()

    print(f"[finished] method={method} folds={len(fold_rows)}", flush=True)


def main(method: str) -> None:
    output_log_dir = log_dir(method)
    output_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_log_dir / (
        "standalone_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".log"
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("x", encoding="utf-8", buffering=1) as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"[log] {log_path}", flush=True)
            run_experiment(method)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

"""MI cross-trial deep-learning runner."""

from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence, TextIO

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from torch.utils.data import DataLoader
from torcheeg import transforms

from experiment_config import (
    CACHE_ROOT,
    DERIVATIVES_ROOT,
    MI_CROSS_TRIAL_CACHE,
)
from hs_stroke.datasets.derivatives import DerivativesDataset
from hs_stroke.models import ATCNet, ArjunViT, EEGConformer, EEGNet, IFNet
from hs_stroke.trainers import ClassifierTrainer
from hs_stroke.splits import CrossTrialSplit


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
SPLIT_CACHE = CACHE_ROOT / "splits" / "mi_cross_trial"

N_SPLITS = 5
EPOCHS = int(os.environ.get("HS_STROKE_EPOCHS", "300"))
BATCH_SIZE = int(os.environ.get("HS_STROKE_BATCH_SIZE", "32"))
NUM_WORKERS = int(os.environ.get("HS_STROKE_NUM_WORKERS", "2"))
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP_VAL = 1.0
EARLY_STOP_PATIENCE = int(os.environ.get("HS_STROKE_EARLY_STOP_PATIENCE", "199"))
RANDOM_SEED = 42
METRICS = ["accuracy", "recall", "precision", "f1score", "kappa"]
OVERLAP = 0.5
SPLIT_PROTOCOL = "cross_trial"
SUPPORTED_MODELS = {"atcnet", "eegnet", "eegconformer", "ifnet", "arjunvit"}


@dataclass(frozen=True)
class CrossTrialSpec:
    data_root: Path = DERIVATIVES_ROOT
    dataset_cache: Path = MI_CROSS_TRIAL_CACHE
    split_cache: Path = SPLIT_CACHE
    overlap: float = OVERLAP


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


class BestValidationMetrics(Callback):
    def __init__(self, metric_names: Sequence[str]) -> None:
        super().__init__()
        self.metric_names = list(metric_names)
        self.best_epoch = None
        self.best_metrics = {}

    def on_validation_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        accuracy = trainer.callback_metrics.get("val_accuracy")
        if accuracy is None:
            return
        current_accuracy = float(accuracy.detach().cpu())
        if current_accuracy <= self.best_metrics.get("accuracy", float("-inf")):
            return

        selected_metrics = {}
        for metric_name in ["loss", *self.metric_names]:
            value = trainer.callback_metrics.get("val_" + metric_name)
            if value is None:
                raise RuntimeError(
                    f"epoch={trainer.current_epoch} missing val_{metric_name}"
                )
            selected_metrics[metric_name] = float(value.detach().cpu())
        self.best_epoch = int(trainer.current_epoch)
        self.best_metrics = selected_metrics


def build_model(model_name: str) -> torch.nn.Module:
    if model_name == "atcnet":
        return ATCNet(
            chunk_size=128,
            num_electrodes=11,
            in_channels=1,
            num_windows=2,
            F1=16,
            D=4,
            num_classes=2,
        )
    if model_name == "eegnet":
        return EEGNet(
            chunk_size=128,
            num_electrodes=11,
            F1=8,
            F2=16,
            D=2,
            kernel_1=64,
            kernel_2=16,
            dropout=0.25,
            num_classes=2,
        )
    if model_name == "eegconformer":
        return EEGConformer(
            num_electrodes=11,
            sampling_rate=128,
            hid_channels=40,
            depth=6,
            heads=10,
            num_classes=2,
        )
    if model_name == "ifnet":
        return IFNet(
            in_planes=11,
            out_planes=96,
            kernel_size=32,
            radix=1,
            patch_size=4,
            time_points=128,
            num_classes=2,
        )
    if model_name == "arjunvit":
        return ArjunViT(
            chunk_size=128,
            t_patch_size=32,
            num_electrodes=11,
            hid_channels=32,
            depth=3,
            heads=4,
            head_channels=64,
            mlp_channels=64,
            num_classes=2,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def build_online_transform(model_name: str):
    if model_name in {"ifnet", "arjunvit"}:
        return transforms.Compose([transforms.ToTensor()])
    return transforms.Compose([transforms.To2d(), transforms.ToTensor()])


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def result_dir(model_name: str, spec: CrossTrialSpec) -> Path:
    return (
        WORKSPACE_ROOT
        / "results"
        / "mi_classification"
        / "cross_trial"
        / model_name
    )


def log_dir(model_name: str, spec: CrossTrialSpec) -> Path:
    return (
        WORKSPACE_ROOT
        / "logs"
        / "mi_classification"
        / "cross_trial"
        / model_name
    )


def write_results(
    model_name: str,
    fold_rows: list[dict],
    spec: CrossTrialSpec,
) -> None:
    output_dir = result_dir(model_name, spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = output_dir / "fold_results.csv"
    subject_csv = output_dir / "subject_results.csv"

    fold_df = pd.DataFrame(fold_rows).sort_values(["subject", "session", "fold"])
    fold_df.to_csv(fold_csv, index=False, float_format="%.6f")

    numeric_columns = ["val_loss", *METRICS]
    subject_df = (
        fold_df.groupby("subject", as_index=False)[numeric_columns]
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
    for column in numeric_columns:
        mean_row[column] = float(subject_df[column].mean())
    subject_df = pd.concat(
        [subject_df, pd.DataFrame([mean_row])],
        ignore_index=True,
    )
    subject_df.to_csv(subject_csv, index=False, float_format="%.6f")


def load_existing_rows(model_name: str, spec: CrossTrialSpec) -> list[dict]:
    fold_csv = result_dir(model_name, spec) / "fold_results.csv"
    if not fold_csv.is_file():
        return []
    fold_df = pd.read_csv(fold_csv)
    if fold_df.empty:
        return []
    if fold_df[["subject", "session", "fold"]].astype(str).duplicated().any():
        raise RuntimeError(f"resume CSV contains duplicate fold keys: {fold_csv}")
    rows = fold_df.to_dict(orient="records")
    print("[resume] loaded_folds=%d file=%s" % (len(rows), fold_csv), flush=True)
    return rows


def sorted_subjects(dataset: DerivativesDataset) -> list[str]:
    return sorted(dataset.info["subject_id"].astype(str).unique().tolist())


def run_experiment(
    model_name: str,
    spec: CrossTrialSpec = CrossTrialSpec(),
) -> None:
    model_name = model_name.lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    torch.set_float32_matmul_precision("high")
    pl.seed_everything(RANDOM_SEED, workers=True)
    if not spec.data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {spec.data_root}")

    dataset_args = {
        "root_path": str(spec.data_root),
        "io_path": str(spec.dataset_cache),
        "num_worker": NUM_WORKERS,
        "online_transform": build_online_transform(model_name),
        "label_transform": transforms.Compose(
            [
                transforms.Select("label"),
                transforms.Lambda(lambda value: value - 1),
            ]
        ),
    }
    dataset = DerivativesDataset(**dataset_args, overlap=spec.overlap)
    data_split = CrossTrialSplit(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_SEED,
        split_path=str(spec.split_cache),
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    subject_count = int(dataset.info["subject_id"].astype(str).nunique())
    trial_count = int(
        dataset.info.groupby(["subject_id", "session_id", "trial_id"]).ngroups
    )
    print(
        "[start] model=%s dataset=%s samples=%d subjects=%d trials=%d "
        "protocol=%s accelerator=%s"
        % (
            model_name,
            spec.data_root,
            len(dataset),
            subject_count,
            trial_count,
            SPLIT_PROTOCOL,
            accelerator,
        ),
        flush=True,
    )
    print(
        "[data] overlap=%.1fs cache=%s split_cache=%s"
        % (spec.overlap, spec.dataset_cache, spec.split_cache),
        flush=True,
    )
    selected_subjects = sorted_subjects(dataset)
    fold_rows = load_existing_rows(model_name, spec)
    completed = {
        (str(row["subject"]), str(row["session"]), int(row["fold"]))
        for row in fold_rows
    }
    for selected_subject in selected_subjects:
        split_iterator = data_split.split(dataset, subject=selected_subject)
        for train_set, val_set, subject, session, fold_id in split_iterator:
            subject = str(subject)
            session = str(session)
            fold = int(fold_id) + 1
            if (subject, session, fold) in completed:
                print(
                    "[resume-skip] model=%s subject=%s session=%s fold=%d"
                    % (model_name, subject, session, fold),
                    flush=True,
                )
                continue
            train_loader = DataLoader(
                train_set,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
            val_loader = DataLoader(
                val_set,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True,
            )
            model = build_model(model_name)
            parameter_count = count_parameters(model)
            best_val_callback = BestValidationMetrics(METRICS)
            early_stop_callback = EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=EARLY_STOP_PATIENCE,
                verbose=False,
            )
            trainer = ClassifierTrainer(
                model=model,
                num_classes=2,
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                label_smoothing=LABEL_SMOOTHING,
                enable_checkpointing=False,
                accelerator=accelerator,
                metrics=METRICS,
            )
            print(
                "[train] model=%s subject=%s session=%s fold=%d/%d "
                "train=%d val=%d batch_size=%d num_workers=%d params=%d"
                % (
                    model_name,
                    subject,
                    session,
                    fold,
                    N_SPLITS,
                    len(train_set),
                    len(val_set),
                    BATCH_SIZE,
                    NUM_WORKERS,
                    parameter_count,
                ),
                flush=True,
            )
            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=EPOCHS,
                callbacks=[best_val_callback, early_stop_callback],
                gradient_clip_val=GRAD_CLIP_VAL,
                enable_progress_bar=False,
                enable_model_summary=False,
                limit_val_batches=1.0,
            )
            if best_val_callback.best_epoch is None:
                raise RuntimeError(
                    f"{model_name} subject={subject} session={session} "
                    f"fold={fold} has no validation metrics"
                )

            row = {
                "subject": subject,
                "session": session,
                "fold": fold,
                "train_samples": len(train_set),
                "val_samples": len(val_set),
                "best_epoch": best_val_callback.best_epoch,
                "val_loss": best_val_callback.best_metrics["loss"],
                "params": parameter_count,
                "evaluation_role": "validation",
                "selection_protocol": "best_val_accuracy_no_checkpoint",
                "split_protocol": SPLIT_PROTOCOL,
            }
            for metric in METRICS:
                row[metric] = best_val_callback.best_metrics[metric]
            fold_rows.append(row)
            write_results(model_name, fold_rows, spec)
            print(
                "[best-val] model=%s subject=%s session=%s fold=%d epoch=%d "
                "accuracy=%.6f f1score=%.6f kappa=%.6f"
                % (
                    model_name,
                    subject,
                    session,
                    fold,
                    row["best_epoch"],
                    row["accuracy"],
                    row["f1score"],
                    row["kappa"],
                ),
                flush=True,
            )

            del trainer, model, train_loader, val_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(
        "[finished] model=%s folds=%d results=%s"
        % (model_name, len(fold_rows), result_dir(model_name, spec)),
        flush=True,
    )


def main(model_name: str, spec: CrossTrialSpec = CrossTrialSpec()) -> None:
    output_log_dir = log_dir(model_name, spec)
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
            run_experiment(model_name, spec)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

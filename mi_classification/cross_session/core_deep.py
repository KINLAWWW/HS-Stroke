"""Cross-session deep-learning runner for MI classification."""

import os
import re
import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torcheeg import transforms

from experiment_config import (
    DERIVATIVES_ROOT,
    MI_CROSS_SESSION_CACHE,
    MI_CROSS_SESSION_SPLIT_ROOT,
    RESULTS_ROOT,
)
from hs_stroke.datasets.derivatives import DerivativesDataset
from hs_stroke.models import ATCNet, ArjunViT, EEGConformer, EEGNet, IFNet
from hs_stroke.splits import CrossSessionSplit, _DatasetInfoView
from hs_stroke.trainers import ClassifierTrainer

SESSION_STRATEGY = "cross_session"


@dataclass
class ExperimentConfig:
    models: tuple[str, ...] = ("atcnet", "eegnet", "eegconformer", "ifnet")
    root_path: str | Path = DERIVATIVES_ROOT
    io_path: str | Path = MI_CROSS_SESSION_CACHE
    results_root: str | Path = RESULTS_ROOT / "mi_classification" / "cross_session"
    split_root: str | Path = MI_CROSS_SESSION_SPLIT_ROOT
    epochs: int = 300
    batch_size: int = 32
    num_workers: int = 0
    val_session_ratio: float = 0.2
    val_min_sessions: int = 1
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip_val: float = 1.0
    early_stop_patience: int = 199
    random_seed: int = 42
    force_resplit: bool = False
    max_subjects: int = 0
    resume: bool = False


def _build_config(overrides: dict) -> ExperimentConfig:
    known_fields = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(set(overrides) - known_fields)
    if unknown:
        raise TypeError(f"Unknown experiment configuration fields: {unknown}")
    return ExperimentConfig(**overrides)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str):
    name = name.lower()
    if name == 'atcnet':
        return ATCNet(chunk_size=128,
                      num_electrodes=11,
                      in_channels=1,
                      num_windows=2,
                      F1=16,
                      D=4,
                      num_classes=2)
    if name == 'eegnet':
        return EEGNet(chunk_size=128,
                      num_electrodes=11,
                      F1=8,
                      F2=16,
                      D=2,
                      kernel_1=64,
                      kernel_2=16,
                      dropout=0.25,
                      num_classes=2)
    if name == 'eegconformer':
        return EEGConformer(num_electrodes=11,
                            sampling_rate=128,
                            hid_channels=40,
                            depth=6,
                            heads=10,
                            num_classes=2)
    if name == 'arjunvit':
        return ArjunViT(chunk_size=128,
                        t_patch_size=32,
                        num_electrodes=11,
                        hid_channels=32,
                        depth=3,
                        heads=4,
                        head_channels=64,
                        mlp_channels=64,
                        num_classes=2)
    if name == 'ifnet':
        return IFNet(in_planes=11,
                     out_planes=96,
                     kernel_size=32,
                     radix=1,
                     patch_size=4,
                     time_points=128,
                     num_classes=2)
    raise ValueError(f'Unknown model: {name}')


def build_dataset(root_path: str, io_path: str, model_name: str, num_workers: int):
    if model_name.lower() in {'ifnet', 'arjunvit'}:
        online_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        online_transform = transforms.Compose([
            transforms.To2d(),
            transforms.ToTensor()
        ])

    return DerivativesDataset(
        io_path=io_path,
        root_path=root_path,
        num_worker=num_workers,
        online_transform=online_transform,
        label_transform=transforms.Compose([
            transforms.Select('label'),
            transforms.Lambda(lambda x: x - 1)
        ])
    )


def _sorted_session_nums(info_df: pd.DataFrame):
    nums = []
    for s in info_df['session_id'].unique().tolist():
        ds = ''.join(ch for ch in str(s) if ch.isdigit())
        if ds:
            nums.append(int(ds))
    return sorted(list(set(nums)))


def _split_train_val_by_session(train_set, val_session_ratio: float, val_min_sessions: int):
    train_info = train_set.info.copy()
    session_nums = train_info['session_id'].apply(lambda s: int(re.findall(r'(\d+)', str(s))[-1]))
    unique_session_nums = sorted(list(set(session_nums.tolist())))

    if len(unique_session_nums) < 2:
        raise ValueError('Not enough sessions in training set to build a validation split.')

    val_count = int(round(len(unique_session_nums) * val_session_ratio))
    val_count = max(val_min_sessions, val_count)
    val_count = min(len(unique_session_nums) - 1, val_count)

    val_session_nums = set(unique_session_nums[-val_count:])
    train_session_nums = set(unique_session_nums[:-val_count])

    core_train_info = train_info[session_nums.isin(train_session_nums)].copy()
    val_info = train_info[session_nums.isin(val_session_nums)].copy()

    if len(core_train_info) == 0 or len(val_info) == 0:
        raise ValueError('Temporal train/val split produced empty subset.')

    core_train_set = _DatasetInfoView(train_set, core_train_info)
    val_set = _DatasetInfoView(train_set, val_info)

    return core_train_set, val_set, sorted(list(train_session_nums)), sorted(list(val_session_nums))


def _extract_epoch_history(metrics_csv: str, metrics):
    df = pd.read_csv(metrics_csv)
    if 'epoch' not in df.columns:
        raise ValueError(f'No epoch column found in {metrics_csv}.')

    epoch_series = pd.to_numeric(df['epoch'], errors='coerce')
    valid_epochs = sorted([int(e) for e in epoch_series.dropna().unique().tolist()])

    rows = []
    for ep in valid_epochs:
        row = {'epoch': int(ep)}
        has_any = False
        for phase in ['train', 'val']:
            for metric in ['loss', *metrics]:
                col = f'{phase}_{metric}'
                if col not in df.columns:
                    continue
                vals = df.loc[(epoch_series == ep) & df[col].notna(), col]
                if len(vals) > 0:
                    row[col] = float(vals.iloc[-1])
                    has_any = True
        if has_any:
            rows.append(row)

    return pd.DataFrame(rows)


def _extract_best_epoch_metrics(epoch_df: pd.DataFrame, best_epoch: int) -> dict:
    best_rows = epoch_df.loc[
        pd.to_numeric(epoch_df['epoch'], errors='coerce') == int(best_epoch)
    ]
    if len(best_rows) == 0:
        raise RuntimeError(f'No validation metrics found for best epoch {best_epoch}.')

    best_row = best_rows.iloc[-1]
    output = {}
    for metric in ['loss', 'accuracy', 'f1score', 'kappa']:
        column = f'val_{metric}'
        if column not in best_row or pd.isna(best_row[column]):
            raise RuntimeError(
                f'Missing {column} for best epoch {best_epoch}.'
            )
        output[f'best_val_{metric}'] = float(best_row[column])
    return output


def _write_subject_results(rows, output_csv: str) -> None:
    """Write subject-level results."""
    subject_df = pd.DataFrame(rows)
    numeric_columns = [
        'core_train_session_count',
        'val_session_count',
        'heldout_session_count',
        'best_epoch',
        'best_val_loss',
        'best_val_accuracy',
        'best_val_f1score',
        'best_val_kappa',
        'test_loss',
        'test_accuracy',
        'test_f1score',
        'test_kappa',
        'params',
    ]
    mean_row = {
        'subject': 'MEAN',
        'model': rows[0]['model'],
        'session_strategy': rows[0]['session_strategy'],
        'selection_metric': 'val_accuracy',
        'best_epoch_metric_role': 'validation',
        'core_train_sessions': '',
        'val_sessions': '',
        'heldout_sessions': '',
    }
    for column in numeric_columns:
        if column not in subject_df.columns:
            subject_df[column] = np.nan
        mean_row[column] = float(pd.to_numeric(subject_df[column]).mean())

    output_df = pd.concat([subject_df, pd.DataFrame([mean_row])], ignore_index=True)
    output_df.to_csv(output_csv, index=False, float_format='%.6f')


def run_experiment(**overrides):
    args = _build_config(overrides)

    results_root = Path(args.results_root)
    torch.set_float32_matmul_precision('high')
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    metrics = ['accuracy', 'f1score', 'kappa']
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    for model_name in args.models:
        model_name = model_name.lower()
        results_dir = results_root / model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        dataset = build_dataset(
            args.root_path,
            args.io_path,
            model_name,
            num_workers=args.num_workers,
        )

        print(f'[info] Using IO cache path: {args.io_path}', flush=True)
        print(f'[info] Dataset size from cache: {len(dataset)}', flush=True)

        tag = model_name
        subject_results_csv = str(results_dir / 'subject_results.csv')
        summary_csv = str(results_dir / 'summary.csv')

        split_path = Path(args.split_root) / tag
        if args.force_resplit and os.path.exists(split_path):
            shutil.rmtree(split_path)

        data_split = CrossSessionSplit(split_path=split_path)

        epoch_rows = []
        subject_rows = []
        completed_subjects = set()
        if args.resume and os.path.isfile(subject_results_csv):
            existing_results = pd.read_csv(subject_results_csv)
            existing_results = existing_results[
                existing_results['subject'].astype(str) != 'MEAN'
            ].copy()
            if existing_results['subject'].astype(str).duplicated().any():
                raise RuntimeError(
                    f'Duplicate subjects in resume CSV: {subject_results_csv}'
                )
            if len(existing_results) > 0:
                unexpected_models = set(existing_results['model'].astype(str)) - {model_name}
                unexpected_strategies = (
                    set(existing_results['session_strategy'].astype(str))
                    - {SESSION_STRATEGY}
                )
                if unexpected_models or unexpected_strategies:
                    raise RuntimeError(
                        'Resume CSV does not match the requested model/session strategy.'
                    )
                subject_rows = existing_results.to_dict(orient='records')
                completed_subjects = set(
                    existing_results['subject'].astype(str).tolist()
                )
                for existing_row in subject_rows:
                    completed_subject = str(existing_row['subject'])
                    metrics_csv = os.path.join(
                        results_dir,
                        'val_logs',
                        tag,
                        completed_subject,
                        'pl',
                        'v0',
                        'metrics.csv',
                    )
                    if not os.path.isfile(metrics_csv):
                        continue
                    epoch_df = _extract_epoch_history(metrics_csv, metrics)
                    if len(epoch_df) == 0:
                        continue
                    existing_row.update(
                        _extract_best_epoch_metrics(
                            epoch_df,
                            int(existing_row['best_epoch']),
                        )
                    )
                    existing_row['selection_metric'] = 'val_accuracy'
                    existing_row['best_epoch_metric_role'] = 'validation'
                    epoch_df['subject'] = completed_subject
                    epoch_df['model'] = model_name
                    epoch_df['session_strategy'] = SESSION_STRATEGY
                    for column in [
                        'core_train_sessions',
                        'val_sessions',
                        'heldout_sessions',
                        'core_train_session_count',
                        'val_session_count',
                        'heldout_session_count',
                        'params',
                    ]:
                        epoch_df[column] = existing_row[column]
                    epoch_rows.append(epoch_df)
                _write_subject_results(subject_rows, subject_results_csv)
                print(
                    f'[resume] completed_subjects={sorted(completed_subjects)} '
                    f'next_subject_count={len(completed_subjects) + 1}',
                    flush=True,
                )
        params_count = count_parameters(build_model(model_name))

        for split_idx, (train_set, test_set, subject) in enumerate(data_split.split(dataset), start=1):
            if args.max_subjects > 0 and split_idx > args.max_subjects:
                break
            if str(subject) in completed_subjects:
                print(f'[resume-skip] subject={subject}', flush=True)
                continue

            original_train_sessions = _sorted_session_nums(train_set.info)
            heldout_sessions = _sorted_session_nums(test_set.info)
            core_train_set, val_set, core_train_sessions, val_sessions = _split_train_val_by_session(
                train_set,
                val_session_ratio=args.val_session_ratio,
                val_min_sessions=args.val_min_sessions
            )

            train_loader = DataLoader(core_train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
            val_loader = DataLoader(val_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=True)
            test_loader = DataLoader(test_set,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     pin_memory=True)

            trainer = ClassifierTrainer(model=build_model(model_name),
                                        num_classes=2,
                                        lr=1e-3 if model_name not in {'eegconformer', 'ifnet'} else 1e-4,
                                        weight_decay=args.weight_decay,
                                        label_smoothing=args.label_smoothing,
                                        enable_checkpointing=True,
                                        metrics=metrics,
                                        accelerator=accelerator)

            logger_dir = os.path.join(results_dir, 'val_logs', tag, str(subject))
            if os.path.exists(logger_dir):
                shutil.rmtree(logger_dir)
            os.makedirs(logger_dir, exist_ok=True)

            early_stop_callback = EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=args.early_stop_patience,
                verbose=False
            )
            checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(logger_dir, 'checkpoints'),
                filename='best-{epoch:03d}-{val_accuracy:.6f}',
                monitor='val_accuracy',
                mode='max',
                save_top_k=1,
                save_last=False,
                save_weights_only=True,
            )
            csv_logger = CSVLogger(save_dir=logger_dir, name='pl', version='v0')

            print(f'[cross-session] model={model_name} subject={subject} split={split_idx} '
                  f'core_train={len(core_train_set)} val={len(val_set)} heldout={len(test_set)} '
                  f'train_sessions={core_train_sessions} val_sessions={val_sessions} heldout_sessions={heldout_sessions} '
                  f'original_train_sessions={original_train_sessions} params={params_count} device={accelerator}',
                  flush=True)

            trainer.fit(train_loader,
                        val_loader,
                        max_epochs=args.epochs,
                        callbacks=[early_stop_callback, checkpoint_callback],
                        logger=csv_logger,
                        gradient_clip_val=args.grad_clip_val,
                        enable_progress_bar=False,
                        enable_model_summary=False,
                        limit_val_batches=1.0)

            metrics_csv = os.path.join(csv_logger.log_dir, 'metrics.csv')
            epoch_df = _extract_epoch_history(metrics_csv, metrics)
            if len(epoch_df) == 0:
                raise RuntimeError(f'No epoch metrics extracted for subject {subject} from {metrics_csv}.')

            epoch_df['subject'] = subject
            epoch_df['model'] = model_name
            epoch_df['session_strategy'] = SESSION_STRATEGY
            epoch_df['core_train_sessions'] = '-'.join(map(str, core_train_sessions))
            epoch_df['val_sessions'] = '-'.join(map(str, val_sessions))
            epoch_df['heldout_sessions'] = '-'.join(map(str, heldout_sessions))
            epoch_df['core_train_session_count'] = len(core_train_sessions)
            epoch_df['val_session_count'] = len(val_sessions)
            epoch_df['heldout_session_count'] = len(heldout_sessions)
            epoch_df['params'] = params_count
            epoch_rows.append(epoch_df)

            if not checkpoint_callback.best_model_path:
                raise RuntimeError(f'No best checkpoint saved for subject {subject}.')

            checkpoint = torch.load(
                checkpoint_callback.best_model_path,
                map_location='cpu',
                weights_only=False,
            )
            trainer.load_state_dict(checkpoint['state_dict'])
            best_epoch = int(checkpoint['epoch'])
            best_val_metrics = _extract_best_epoch_metrics(epoch_df, best_epoch)
            checkpoint_best_val_accuracy = float(
                checkpoint_callback.best_model_score.detach().cpu()
            )
            if not np.isclose(
                best_val_metrics['best_val_accuracy'],
                checkpoint_best_val_accuracy,
                atol=1e-6,
            ):
                raise RuntimeError(
                    'Best checkpoint score does not match the saved epoch metrics: '
                    f"{checkpoint_best_val_accuracy} vs "
                    f"{best_val_metrics['best_val_accuracy']}"
                )
            test_output = trainer.test(
                test_loader,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            if len(test_output) != 1:
                raise RuntimeError(f'Unexpected test output for subject {subject}: {test_output}')
            test_metrics = test_output[0]

            subject_row = {
                'subject': subject,
                'model': model_name,
                'session_strategy': SESSION_STRATEGY,
                'core_train_sessions': '-'.join(map(str, core_train_sessions)),
                'val_sessions': '-'.join(map(str, val_sessions)),
                'heldout_sessions': '-'.join(map(str, heldout_sessions)),
                'core_train_session_count': len(core_train_sessions),
                'val_session_count': len(val_sessions),
                'heldout_session_count': len(heldout_sessions),
                'best_epoch': best_epoch,
                'selection_metric': 'val_accuracy',
                'best_epoch_metric_role': 'validation',
                **best_val_metrics,
                'test_loss': float(test_metrics['test_loss']),
                'test_accuracy': float(test_metrics['test_accuracy']),
                'test_f1score': float(test_metrics['test_f1score']),
                'test_kappa': float(test_metrics['test_kappa']),
                'params': params_count,
            }
            subject_rows.append(subject_row)
            _write_subject_results(subject_rows, subject_results_csv)
            print(
                f"[test] subject={subject} best_epoch={best_epoch} "
                f"best_val_loss={subject_row['best_val_loss']:.6f} "
                f"best_val_accuracy={subject_row['best_val_accuracy']:.6f} "
                f"best_val_f1score={subject_row['best_val_f1score']:.6f} "
                f"best_val_kappa={subject_row['best_val_kappa']:.6f} "
                f"test_accuracy={subject_row['test_accuracy']:.6f} "
                f"test_f1score={subject_row['test_f1score']:.6f} "
                f"test_kappa={subject_row['test_kappa']:.6f}",
                flush=True,
            )

            del checkpoint, test_output, test_metrics
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if len(epoch_rows) == 0:
            raise RuntimeError(f'No subject split was generated for model {model_name}.')

        epoch_metrics_df = pd.concat(epoch_rows, ignore_index=True)
        epoch_metrics_df = epoch_metrics_df.sort_values(['subject', 'epoch']).reset_index(drop=True)
        summary_agg = {'subject': 'count'}
        for phase in ['train', 'val']:
            for metric in ['loss', *metrics]:
                col = f'{phase}_{metric}'
                if col in epoch_metrics_df.columns:
                    summary_agg[col] = 'mean'

        epoch_summary_df = epoch_metrics_df.groupby('epoch', as_index=False).agg(summary_agg)
        epoch_summary_df = epoch_summary_df.rename(columns={'subject': 'num_subjects'})
        best_idx = epoch_summary_df['val_accuracy'].astype(float).idxmax()
        best_epoch = int(epoch_summary_df.loc[best_idx, 'epoch'])
        best_value = float(epoch_summary_df.loc[best_idx, 'val_accuracy'])
        print(f'[global-best] model={model_name} best_epoch={best_epoch} avg_val_accuracy={best_value:.6f}', flush=True)
        completed_df = pd.DataFrame(subject_rows)
        summary_rows = []
        for output_metric, source_column in [
            ('accuracy', 'test_accuracy'),
            ('f1score', 'test_f1score'),
            ('kappa', 'test_kappa'),
        ]:
            values = pd.to_numeric(completed_df[source_column], errors='coerce').dropna()
            summary_rows.append({
                'metric': output_metric,
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'count': int(len(values)),
            })
        pd.DataFrame(summary_rows).to_csv(
            summary_csv,
            index=False,
            float_format='%.6f',
        )
        shutil.rmtree(os.path.join(results_dir, 'val_logs', tag), ignore_errors=True)
        print(
            f'[done] {model_name}: subject_results -> {subject_results_csv}; '
            f'summary -> {summary_csv}',
            flush=True,
        )

"""LOSO deep-learning runner for FMA regression."""

import os
import re
import sys
from copy import copy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torcheeg import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hs_stroke.datasets.fma import FMADataset
from hs_stroke.models import ATCNet_R, ArjunViT_R, IFNet_R
from hs_stroke.splits import LeaveOneSubjectOut
from hs_stroke.trainers import RegressorTrainer
from experiment_config import (
    FMA_DATASET_CACHE,
    FMA_SESSIONS_TSV,
    FMA_SOURCE_ROOT,
    FMA_SPLIT_ROOT,
    PARTICIPANTS_TSV,
    RESULTS_ROOT,
)

FIXED_SCALE = 66.0


def _infer_subject_id_from_trial_id(trial_id: str) -> str:
    matched = re.search(r'(sub[-_]\d+)', str(trial_id))
    if matched is None:
        raise ValueError(f'Cannot infer subject_id from trial_id: {trial_id}')
    return matched.group(1).replace('_', '-')


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(name: str):
    name = name.lower()
    if name == 'ifnet':
        return IFNet_R(
            in_planes=11,
            out_planes=96,
            kernel_size=3,
            radix=1,
            patch_size=32,
            dropout_fc=0.5,
        )
    if name == 'atcnet':
        return ATCNet_R(
            chunk_size=128,
            num_electrodes=11,
            in_channels=1,
            num_windows=2,
            F1=16,
            D=4,
            tcn_kernel_size=4,
            tcn_depth=2,
            conv_pool_size=7,
            output_dim=1,
        )
    if name == 'arjunvit':
        return ArjunViT_R(
            num_electrodes=11,
            chunk_size=128,
            t_patch_size=32,
            hid_channels=32,
            depth=3,
            heads=4,
            head_channels=64,
            mlp_channels=64,
            num_outputs=1,
            dropout=0.1,
        )
    raise ValueError(f'Unknown model: {name}')


def _split_train_val_by_ratio(train_set, val_ratio: float, random_state: int):
    info = train_set.info.copy().reset_index(drop=True)
    if len(info) < 2:
        raise ValueError('Not enough training samples to create a validation split.')

    val_count = max(1, int(round(len(info) * val_ratio)))
    val_count = min(len(info) - 1, val_count)

    val_info = info.sample(n=val_count, random_state=random_state, replace=False)
    train_info = info.drop(val_info.index).reset_index(drop=True)
    val_info = val_info.reset_index(drop=True)

    core_train_set = copy(train_set)
    core_train_set.info = train_info

    val_set = copy(train_set)
    val_set.info = val_info

    return core_train_set, val_set


def _summarize(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        vals = df[metric].astype(float).values
        if metric == 'mae':
            rows.append({'metric': metric, 'mean': float(vals.mean()), 'std': float(vals.std(ddof=1) if len(vals) > 1 else 0.0)})
        elif metric == 'mse':
            rows.append({'metric': metric, 'mean': float(vals.mean()), 'std': float(vals.std(ddof=1) if len(vals) > 1 else 0.0)})
    return pd.DataFrame(rows)


def run_experiment(
    *,
    models=("ifnet", "atcnet", "arjunvit"),
    root_path=FMA_SOURCE_ROOT,
    participants_path=PARTICIPANTS_TSV,
    session_map_path=FMA_SESSIONS_TSV,
    io_path=FMA_DATASET_CACHE,
    split_path=FMA_SPLIT_ROOT / "loso",
    results_root=RESULTS_ROOT / "fma_regression" / "loso",
    val_ratio=0.1,
    epochs=100,
    batch_size=32,
    num_workers=0,
    random_state=42,
    max_subjects=0,
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    torch.set_float32_matmul_precision('high')

    results_root = Path(results_root)
    dataset = FMADataset(
        io_path=io_path,
        root_path=root_path,
        participants_path=participants_path,
        session_map_path=session_map_path,
        online_transform=transforms.Compose([
            transforms.ToTensor()
        ]),
        label_transform=transforms.Compose([
            transforms.Select('label')
        ])
    )

    if 'subject_id' not in dataset.info.columns:
        if 'trial_id' not in dataset.info.columns:
            raise KeyError('dataset.info does not contain subject_id or trial_id.')
        dataset.info = dataset.info.copy()
        dataset.info['subject_id'] = dataset.info['trial_id'].apply(_infer_subject_id_from_trial_id)
        print('[info] subject_id column was inferred from trial_id for cached data.', flush=True)

    print(f'[info] dataset size: {len(dataset)}', flush=True)
    print('[info] LOSO holds out each complete subject.', flush=True)

    metrics = ['mae', 'mse']
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    for model_name in models:
        model_name = model_name.lower()
        results_dir = results_root / model_name
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv = results_dir / 'predictions.csv'
        summary_csv = results_dir / 'summary.csv'

        if os.path.exists(results_csv):
            os.remove(results_csv)

        model_split_path = str(Path(split_path) / model_name)
        data_split = LeaveOneSubjectOut(split_path=model_split_path)

        rows = []
        subject_count = 0
        for fold_idx, (train_set, test_set, subject) in enumerate(data_split.split(dataset)):
            subject_count += 1
            if max_subjects > 0 and subject_count > max_subjects:
                break

            train_set, val_set = _split_train_val_by_ratio(train_set, val_ratio, random_state)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            model = build_model(model_name)
            trainer = RegressorTrainer(
                model=model,
                lr=1e-4,
                metrics=metrics,
                devices=1,
                accelerator=accelerator,
            )

            print(
                f'[train] model={model_name} subject={subject} fold={fold_idx + 1} '
                f'train={len(train_set)} val={len(val_set)} test={len(test_set)} params={count_parameters(model)}',
                flush=True,
            )

            trainer.fit(
                train_loader,
                val_loader,
                max_epochs=epochs,
                limit_val_batches=1.0,
                enable_model_summary=False,
            )

            test_result = trainer.test(test_loader)[0]
            preds = trainer.predict(test_loader)
            y_pred = torch.cat(preds, dim=0).squeeze(-1).cpu().numpy() * FIXED_SCALE
            y_true = np.concatenate([y.cpu().numpy() for _, y in test_loader], axis=0) * FIXED_SCALE

            row = {
                'subject': subject,
                'fold': fold_idx + 1,
                'params': int(count_parameters(model)),
                'mae': float(test_result['test_mae'] * FIXED_SCALE),
                'mse': float(test_result['test_mse'] * (FIXED_SCALE ** 2)),
                'true_fma': float(y_true.mean()),
                'predicted_fma': float(y_pred.mean()),
            }
            rows.append(row)

            pd.DataFrame(rows).to_csv(results_csv, index=False, float_format='%.6f')

        if len(rows) == 0:
            raise RuntimeError(f'No results produced for model {model_name}.')

        summary_df = _summarize(pd.DataFrame(rows), metrics)
        summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
        print(f'[done] {model_name} -> {results_csv} | {summary_csv}', flush=True)

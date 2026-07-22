"""Session-level Ridge baseline runner for FMA regression under LOSO."""

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hs_stroke.models.regression.sbp_ridge import fit_ridge_predict, load_samples
from experiment_config import FMA_SESSIONS_TSV, FMA_SOURCE_ROOT, PARTICIPANTS_TSV, RESULTS_ROOT


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_experiment(
    *,
    root_path=FMA_SOURCE_ROOT,
    participants_path=PARTICIPANTS_TSV,
    session_map_path=FMA_SESSIONS_TSV,
    results_dir=RESULTS_ROOT / "fma_regression" / "loso" / "sbp_ridge",
    alpha_grid=(0.1, 1.0, 10.0),
):
    _log('[step] start LOSO Ridge baseline')
    _log(f'[config] root_path={root_path}')
    _log(f'[config] results_dir={results_dir}')
    _log(f'[config] alpha_grid={alpha_grid}')

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_csv = results_dir / 'predictions.csv'
    summary_csv = results_dir / 'summary.csv'

    samples = load_samples(
        root_path,
        participants_path,
        session_map_path,
    )
    if len(samples) == 0:
        raise RuntimeError(f'No valid samples found in {root_path}.')

    X = np.stack([s.x for s in samples], axis=0)
    y = np.asarray([s.fma for s in samples], dtype=np.float64)
    subjects = np.asarray([s.subject_id for s in samples])
    sessions = np.asarray([s.session_id for s in samples])

    _log(f'[data] X_shape={X.shape} y_shape={y.shape} unique_subjects={len(np.unique(subjects))}')
    subj_counts = pd.Series(subjects).value_counts().sort_index()
    _log('[data] per-subject session counts: ' + ', '.join([f'{k}:{int(v)}' for k, v in subj_counts.items()]))

    rows: List[Dict] = []
    all_subjects = sorted(np.unique(subjects))
    _log(f'[split] total LOSO folds={len(all_subjects)} (one subject held out each fold)')

    for fold_no, test_subject in enumerate(all_subjects, start=1):
        te_idx = np.where(subjects == test_subject)[0]
        tr_idx = np.where(subjects != test_subject)[0]

        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        gtr = subjects[tr_idx]

        same_subject_in_train = (subjects[tr_idx] == test_subject)
        _log(
            f'[split {fold_no:02d}/{len(all_subjects)}] '
            f'test_subject={test_subject} '
            f'train_n={len(tr_idx)} test_n={len(te_idx)} '
            f'test_sessions={list(sessions[te_idx])} '
            f'same_subject_train_n={int(same_subject_in_train.sum())}'
        )

        pred, best_alpha = fit_ridge_predict(Xtr, ytr, Xte, gtr, list(alpha_grid))
        _log(f'[split {fold_no:02d}/{len(all_subjects)}] selected_alpha={best_alpha}')

        fold_mae = mean_absolute_error(yte, pred)
        fold_mse = mean_squared_error(yte, pred)
        _log(
            f'[split {fold_no:02d}/{len(all_subjects)}] '
            f'fold_mae={fold_mae:.4f} fold_mse={fold_mse:.4f}'
        )

        for i, idx in enumerate(te_idx):
            rows.append({
                'fold': fold_no,
                'subject_id': subjects[idx],
                'session_id': sessions[idx],
                'alpha': best_alpha,
                'true_fma': float(y[idx]),
                'predicted_fma': float(pred[i]),
                'abs_error': float(abs(pred[i] - y[idx])),
                'squared_error': float((pred[i] - y[idx]) ** 2),
            })

    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_csv, index=False, float_format='%.6f')

    fold_metrics = (
        pred_df.groupby('fold', sort=True)
        .agg(mae=('abs_error', 'mean'), mse=('squared_error', 'mean'))
        .reset_index()
    )
    summary_df = pd.DataFrame([
        {
            'metric': metric,
            'mean': float(fold_metrics[metric].mean()),
            'std': float(fold_metrics[metric].std(ddof=1)),
        }
        for metric in ('mae', 'mse')
    ])
    summary_df.to_csv(summary_csv, index=False, float_format='%.6f')

    _log(f'[done] predictions: {output_csv}')
    _log(f'[done] summary: {summary_csv}')
    _log(
        f"[metrics] MAE={summary_df.loc[summary_df['metric'] == 'mae', 'mean'].iloc[0]:.4f}, "
        f"MSE={summary_df.loc[summary_df['metric'] == 'mse', 'mean'].iloc[0]:.4f}"
    )

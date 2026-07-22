"""Session-level bandpower Ridge baseline for FMA regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.signal import welch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from hs_stroke.datasets.fma import load_fma_assessments


DEFAULT_CHANNEL_LIST = [
    "FC3", "FC4", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP3", "CP4"
]


@dataclass
class SessionSample:
    subject_id: str
    session_id: str
    fma: float
    x: np.ndarray


def _log(msg: str) -> None:
    print(msg, flush=True)


def _bandpower_1d(signal_1d: np.ndarray, sfreq: int, fmin: float, fmax: float) -> float:
    freqs, psd = welch(signal_1d, fs=sfreq, nperseg=min(len(signal_1d), sfreq))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if mask.sum() == 0:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _extract_session_feature_from_mat(
    mat_file: str,
    sfreq: int = 256,
    mi_start_s: float = 9.0,
    mi_end_s: float = 13.0,
    mu_band: Tuple[float, float] = (8.0, 13.0),
    beta_band: Tuple[float, float] = (13.0, 30.0),
) -> np.ndarray:
    data = loadmat(mat_file, verify_compressed_data_integrity=False)
    eeg = data["EEGdata"].transpose(2, 0, 1)
    cfg = data["configuration_channel"][0]
    ch_names = [ch[0].tolist()[0] for ch in cfg if ch[1].sum()]
    if ch_names != DEFAULT_CHANNEL_LIST:
        raise ValueError(f"Channel mismatch in {mat_file}: {ch_names}")

    start = int(mi_start_s * sfreq)
    end = int(mi_end_s * sfreq)
    eeg = eeg[:, :, start:end]

    n_trials, n_ch, _ = eeg.shape
    feat_trials = np.zeros((n_trials, n_ch * 2), dtype=np.float64)
    for trial_idx in range(n_trials):
        row = []
        for channel_idx in range(n_ch):
            signal = eeg[trial_idx, channel_idx]
            mu_pow = _bandpower_1d(signal, sfreq=sfreq, fmin=mu_band[0], fmax=mu_band[1])
            beta_pow = _bandpower_1d(signal, sfreq=sfreq, fmin=beta_band[0], fmax=beta_band[1])
            row.extend([mu_pow, beta_pow])
        feat_trials[trial_idx] = np.asarray(row, dtype=np.float64)

    return feat_trials.mean(axis=0)


def load_samples(
    root_path: str,
    participants_path: str,
    session_map_path: str,
) -> List[SessionSample]:
    assessments = load_fma_assessments(
        sourcedata_root=root_path,
        participants_path=participants_path,
        session_map_path=session_map_path,
    )
    samples: List[SessionSample] = []
    for assessment in assessments:
        session_feats = [
            _extract_session_feature_from_mat(str(mat_file))
            for mat_file in assessment.mat_files
        ]
        x = np.mean(np.stack(session_feats, axis=0), axis=0)
        samples.append(SessionSample(
            subject_id=assessment.participant_id,
            session_id=assessment.fma_session_id,
            fma=assessment.fma_score,
            x=x,
        ))
        _log(
            f"[load] subject={assessment.participant_id} "
            f"session={assessment.fma_session_id} "
            f"fma={assessment.fma_score:.1f} "
            f"mats={len(assessment.mat_files)} feat_dim={x.shape[0]}"
        )

    _log(f"[load] total session samples={len(samples)}")
    return samples


def choose_alpha_nested_loso(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alpha_grid: List[float],
) -> float:
    unique_groups = np.unique(groups)
    if len(unique_groups) <= 1:
        return alpha_grid[0]

    gkf = GroupKFold(n_splits=len(unique_groups))
    best_alpha = alpha_grid[0]
    best_score = np.inf

    for alpha in alpha_grid:
        fold_mae = []
        for tr_idx, va_idx in gkf.split(X, y, groups=groups):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X[tr_idx])
            Xva = scaler.transform(X[va_idx])
            model = Ridge(alpha=alpha)
            model.fit(Xtr, y[tr_idx])
            pred = model.predict(Xva)
            fold_mae.append(mean_absolute_error(y[va_idx], pred))

        score = float(np.mean(fold_mae))
        if score < best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def fit_ridge_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    groups: np.ndarray,
    alpha_grid: List[float],
) -> Tuple[np.ndarray, float]:
    best_alpha = choose_alpha_nested_loso(X_train, y_train, groups, alpha_grid)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = Ridge(alpha=best_alpha)
    model.fit(X_train_scaled, y_train)
    return model.predict(X_test_scaled), best_alpha

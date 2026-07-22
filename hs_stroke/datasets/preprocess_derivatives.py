"""Create derivative MAT files from source EEG sessions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import mne
import numpy as np
import scipy.io as scio

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_config import DERIVATIVES_ROOT, SOURCE_DATA_ROOT


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
DEFAULT_SAMPLING_RATE = 256
MI_START_SECONDS = 9
MI_END_SECONDS = 13
NOTCH_FREQUENCY = 50.0
DEFAULT_L_FREQ = 8.0
DEFAULT_H_FREQ = 40.0


def detect_and_interpolate_artifacts(
    raw: mne.io.RawArray,
    z_threshold: float = 4.0,
    min_duration_seconds: float = 0.05,
    clip_limit_uv: float = 500.0,
) -> mne.io.RawArray:
    """Interpolate large transient artifacts using simple z-score detection."""

    data = raw.get_data()
    n_times = data.shape[1]
    bad_segments = np.zeros(n_times, dtype=bool)

    for channel_signal in data:
        z_scores = np.abs((channel_signal - np.mean(channel_signal)) / (np.std(channel_signal) + 1e-9))
        bad_indices = np.where(z_scores > z_threshold)[0]
        if len(bad_indices) == 0:
            continue
        for segment in np.split(bad_indices, np.where(np.diff(bad_indices) > 1)[0] + 1):
            duration = len(segment) / raw.info["sfreq"]
            if duration >= min_duration_seconds:
                bad_segments[segment] = True

    if not bad_segments.any():
        raw._data = np.clip(data, -clip_limit_uv, clip_limit_uv)
        return raw

    full_index = np.arange(n_times)
    bad_index = np.where(bad_segments)[0]
    good_index = np.setdiff1d(full_index, bad_index)

    for channel_id in range(data.shape[0]):
        data[channel_id] = np.interp(full_index, good_index, data[channel_id, good_index])

    raw._data = np.clip(data, -clip_limit_uv, clip_limit_uv)
    return raw


def iter_session_mat_files(source_root: Path) -> Iterable[tuple[str, str, Path]]:
    for subject_dir in sorted(source_root.glob("sub-*")):
        if not subject_dir.is_dir():
            continue
        for session_dir in sorted(subject_dir.glob("ses-*")):
            eeg_dir = session_dir / "eeg"
            if not eeg_dir.is_dir():
                continue
            for mat_file in sorted(eeg_dir.glob("*.mat")):
                yield subject_dir.name, session_dir.name, mat_file


def configuration_to_channel_names(configuration_channel: np.ndarray) -> List[str]:
    return [channel[0].tolist()[0] for channel in configuration_channel[0] if channel[1].sum()]


def build_info() -> mne.Info:
    info = mne.create_info(
        ch_names=[name.lower() for name in DEFAULT_CHANNEL_LIST],
        sfreq=DEFAULT_SAMPLING_RATE,
        ch_types=["eeg"] * len(DEFAULT_CHANNEL_LIST),
    )
    montage = mne.channels.make_standard_montage("standard_1020")
    montage.ch_names = [name.lower() for name in montage.ch_names]
    info.set_montage(montage)
    return info


def preprocess_trial(trial: np.ndarray, info: mne.Info) -> np.ndarray:
    raw = mne.io.RawArray(trial * 1e6, info, verbose=False)
    raw.notch_filter(freqs=[NOTCH_FREQUENCY], verbose=False)
    raw.filter(l_freq=DEFAULT_L_FREQ, h_freq=DEFAULT_H_FREQ, verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw = detect_and_interpolate_artifacts(raw)
    start = int(MI_START_SECONDS * DEFAULT_SAMPLING_RATE)
    end = int(MI_END_SECONDS * DEFAULT_SAMPLING_RATE)
    return raw.get_data()[:, start:end]


def preprocess_session(mat_files: List[Path], info: mne.Info) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trials: List[np.ndarray] = []
    labels: List[int] = []
    configuration_channel = None

    for mat_file in mat_files:
        mat = scio.loadmat(mat_file, verify_compressed_data_integrity=False)
        eeg_data = mat["EEGdata"].transpose(2, 0, 1)
        eeg_labels = mat["EEGdatalabel"][:, 0]
        configuration_channel = mat["configuration_channel"]
        channel_names = configuration_to_channel_names(configuration_channel)
        if channel_names != DEFAULT_CHANNEL_LIST:
            raise ValueError(f"Unexpected channel configuration in {mat_file}: {channel_names}")

        for trial, label in zip(eeg_data, eeg_labels):
            trials.append(preprocess_trial(trial, info))
            labels.append(int(label))

    if not trials or configuration_channel is None:
        raise RuntimeError("No valid trials were collected for the session.")

    return np.stack(trials, axis=0), np.asarray(labels, dtype=np.int64).reshape(-1, 1), configuration_channel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate derivative MI MAT files from sourcedata.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DERIVATIVES_ROOT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    info = build_info()

    grouped_files: dict[tuple[str, str], List[Path]] = {}
    for subject_id, session_id, mat_file in iter_session_mat_files(source_root):
        grouped_files.setdefault((subject_id, session_id), []).append(mat_file)

    for (subject_id, session_id), mat_files in sorted(grouped_files.items()):
        eeg_data, eeg_labels, configuration_channel = preprocess_session(mat_files, info)
        save_dir = output_root / subject_id / session_id / "eeg"
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / f"{subject_id}_{session_id}_task-MI_eeg.mat"
        scio.savemat(
            output_file,
            {
                "EEGdata": eeg_data,
                "EEGdatalabel": eeg_labels,
                "configuration_channel": configuration_channel,
            },
        )
        print(f"[saved] {output_file} shape={eeg_data.shape}", flush=True)


if __name__ == "__main__":
    main()

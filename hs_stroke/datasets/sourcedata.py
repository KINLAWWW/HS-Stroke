"""Source MAT dataset loader."""

import os
import shutil
import time
import warnings
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, Union

import autoreject
import mne
import numpy as np
from scipy.io import loadmat
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

DEFAULT_CHANNEL_LIST = [
    "FC3", "FC4", "C5", "C3", "C1", "CZ",
    "C2", "C4", "C6", "CP3", "CP4"
]
DEFAULT_SAMPLING_RATE = 256
CACHE_COMPLETE_MARKER = ".cache_complete"


def cache_complete_marker(io_path: str) -> str:
    return os.path.join(io_path, CACHE_COMPLETE_MARKER)


def remove_incomplete_cache(io_path: Optional[str],
                            retries: int = 5,
                            delay: float = 0.2,
                            require_marker: bool = True) -> None:
    if not io_path or not os.path.exists(io_path):
        return

    if require_marker and os.path.exists(cache_complete_marker(io_path)):
        return

    last_error: Optional[OSError] = None
    for _ in range(retries):
        try:
            shutil.rmtree(io_path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(delay)

    if last_error is not None:
        raise last_error


def mark_cache_complete(io_path: Optional[str]) -> None:
    if not io_path:
        return

    os.makedirs(io_path, exist_ok=True)
    marker_path = cache_complete_marker(io_path)
    with open(marker_path, "w", encoding="ascii"):
        pass


def apply_autoreject(
    epochs,
    verbose: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    n_epochs_before = len(epochs)
    if n_epochs_before < 2:
        return epochs, {
            "n_epochs_before": int(n_epochs_before),
            "n_epochs_after": int(n_epochs_before),
            "n_removed": 0,
            "skipped_autoreject": True,
            "kept_epoch_indices": list(range(n_epochs_before)),
        }

    rejector = autoreject.AutoReject(cv=min(10, n_epochs_before), verbose=verbose)
    epochs, reject_log = rejector.fit_transform(epochs, return_log=True)
    n_epochs_after = len(epochs)
    return epochs, {
        "n_epochs_before": int(n_epochs_before),
        "n_epochs_after": int(n_epochs_after),
        "n_removed": int(n_epochs_before - n_epochs_after),
        "skipped_autoreject": False,
        "kept_epoch_indices": np.flatnonzero(~reject_log.bad_epochs).tolist(),
    }


def normalize_clips(clips: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if clips.size == 0:
        return clips

    clip_min = clips.min(axis=0)
    clip_max = clips.max(axis=0)
    return (clips - clip_min) / (clip_max - clip_min + eps)


class SourceDataset(BaseDataset):
    def __init__(
        self,
        root_path: str,
        duration: int = 1,
        sampling_rate: int = 128,
        l_freq: float = 8.0,
        h_freq: float = 40.0,
        num_channel: int = 11,
        online_transform: Optional[Callable] = None,
        offline_transform: Optional[Callable] = None,
        label_transform: Optional[Callable] = None,
        before_trial: Optional[Callable] = None,
        io_path: Optional[str] = None,
        io_size: int = 1048576,
        io_mode: str = "lmdb",
        num_worker: int = 0,
        verbose: bool = True,
        **extra_params: Any,
    ):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix="datasets")

        cache_is_complete = (
            io_mode != "memory"
            and os.path.isfile(cache_complete_marker(io_path))
        )
        if "trial_offsets" not in extra_params and not cache_is_complete:
            extra_params["trial_offsets"] = self.build_trial_offsets(root_path)

        params = dict(
            root_path=root_path,
            duration=duration,
            sampling_rate=sampling_rate,
            l_freq=l_freq,
            h_freq=h_freq,
            num_channel=num_channel,
            online_transform=online_transform,
            offline_transform=offline_transform,
            label_transform=label_transform,
            before_trial=before_trial,
            io_path=io_path,
            io_size=io_size,
            io_mode=io_mode,
            num_worker=num_worker,
            verbose=verbose,
        )
        params.update(extra_params)

        if io_mode != "memory":
            remove_incomplete_cache(io_path)

        try:
            super().__init__(**params)
        except Exception:
            if io_mode != "memory":
                try:
                    remove_incomplete_cache(io_path, require_marker=False)
                except OSError:
                    pass
            raise

        if io_mode != "memory":
            mark_cache_complete(io_path)

        self.__dict__.update(params)

    @staticmethod
    def process_record(file: Any = None,
                   duration: int = 1,
                   sampling_rate: int = 128,
                   l_freq: float = 8.0,
                   h_freq: float = 40.0,
                   num_channel: int = 11,
                   before_trial: Union[None, Callable] = None,
                   offline_transform: Union[None, Callable] = None,
                   trial_offsets: Optional[Dict[str, int]] = None,
                   **kwargs):
        mat = loadmat(file, verify_compressed_data_integrity=False)
        eeg_data = mat["EEGdata"].transpose(2, 0, 1)
        eeg_labels = mat["EEGdatalabel"][:, 0]

        ch_names = [
            ch[0].tolist()[0]
            for ch in mat["configuration_channel"][0] if ch[1].sum()
        ]
        assert ch_names == DEFAULT_CHANNEL_LIST, f"Incorrect channel list: {ch_names}"

        ch_types = ["eeg"] * len(ch_names)
        ch_names_lower = [c.lower() for c in ch_names]
        info_mne = mne.create_info(ch_names=ch_names_lower,
                                   sfreq=DEFAULT_SAMPLING_RATE,
                                   ch_types=ch_types)
        montage = mne.channels.make_standard_montage("standard_1020")
        montage.ch_names = [ch_name.lower() for ch_name in montage.ch_names]

        mat_path = os.path.abspath(file)
        subject_id = os.path.basename(
            os.path.dirname(os.path.dirname(os.path.dirname(mat_path)))
        )
        session_id = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
        file_id = os.path.splitext(os.path.basename(mat_path))[0]
        if trial_offsets is None or mat_path not in trial_offsets:
            raise ValueError(f"Missing session-level trial offset for {mat_path}")
        trial_offset = int(trial_offsets[mat_path])

        for run_id, (run, label) in enumerate(zip(eeg_data, eeg_labels)):
            start, end = 9 * DEFAULT_SAMPLING_RATE, 13 * DEFAULT_SAMPLING_RATE
            run = run[:, start:end]

            raw = mne.io.RawArray(run, info_mne)
            raw.set_montage(montage)
            raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw = raw.resample(sampling_rate)

            epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True)
            epochs, reject_stats = apply_autoreject(epochs, verbose=False)

            clips = epochs.get_data()
            if clips.size == 0:
                continue
            clips = normalize_clips(clips)

            if before_trial:
                clips = before_trial(clips)

            trial_id = trial_offset + run_id
            trial_key = f"{file_id}_trial-{run_id:03d}"
            kept_window_ids = reject_stats["kept_epoch_indices"]
            for clip, window_id in zip(clips, kept_window_ids):
                clip_id = f"{trial_key}_window-{int(window_id):02d}"
                record_info = dict(
                    clip_id=clip_id,
                    subject_id=subject_id,
                    session_id=session_id,
                    file=file_id,
                    trial_id=trial_id,
                    run_id=run_id,
                    window_id=int(window_id),
                    label=label,
                    start_at=int(window_id) * duration * sampling_rate,
                    end_at=(int(window_id) + 1) * duration * sampling_rate,
                    autoreject=True,
                )

                if offline_transform:
                    clip = offline_transform(eeg=clip[:num_channel])["eeg"]

                yield {"eeg": clip, "key": clip_id, "info": record_info}

    @classmethod
    def build_trial_offsets(cls, root_path: str) -> Dict[str, int]:
        records = cls.collect_records(root_path)
        next_trial_id: Dict[Tuple[str, str], int] = {}
        offsets: Dict[str, int] = {}
        for record in records:
            mat_path = os.path.abspath(record)
            subject_id = os.path.basename(
                os.path.dirname(os.path.dirname(os.path.dirname(mat_path)))
            )
            session_id = os.path.basename(os.path.dirname(os.path.dirname(mat_path)))
            session_key = (subject_id, session_id)
            offsets[mat_path] = next_trial_id.get(session_key, 0)
            labels = loadmat(
                mat_path,
                variable_names=["EEGdatalabel"],
                verify_compressed_data_integrity=False,
            )["EEGdatalabel"]
            next_trial_id[session_key] = offsets[mat_path] + int(labels.shape[0])
        return offsets

    @staticmethod
    def collect_records(root_path: str) -> list:
        records = []
        for subject in os.listdir(root_path):
            spath = os.path.join(root_path, subject)
            if not os.path.isdir(spath):
                continue
            for session in os.listdir(spath):
                ses_path = os.path.join(spath, session, "eeg")
                if os.path.isdir(ses_path):
                    for f in os.listdir(ses_path):
                        if f.endswith(".mat"):
                            records.append(os.path.join(ses_path, f))
        return sorted(records)

    def set_records(self, root_path: str, **kwargs) -> list:
        return self.collect_records(root_path)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        info = self.read_info(index)
        eeg_index = str(info["clip_id"])
        record_id = str(info["_record_id"])

        eeg = self.read_eeg(record_id, eeg_index)
        label = info

        if self.online_transform:
            eeg = self.online_transform(eeg=eeg)["eeg"]

        if self.label_transform:
            label = self.label_transform(y=info)["y"]

        return eeg, label

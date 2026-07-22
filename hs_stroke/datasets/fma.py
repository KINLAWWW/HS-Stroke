"""FMA regression dataset view."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import mne
import numpy as np
from scipy.io import loadmat
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

from hs_stroke.datasets.sourcedata import apply_autoreject, normalize_clips

mne.set_log_level("CRITICAL")
warnings.filterwarnings("ignore")

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
FMA_SCALE = 66.0


@dataclass(frozen=True)
class FMAAssessment:
    participant_id: str
    assessment_id: str
    fma_session_id: str
    fma_score: float
    source_session_ids: tuple[str, ...]
    mat_files: tuple[Path, ...]


def _read_tsv(
    path: Union[str, Path],
    required_columns: set[str],
) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = set(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    missing = required_columns.difference(fieldnames)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return rows


def load_fma_assessments(
    sourcedata_root: Union[str, Path],
    participants_path: Union[str, Path],
    session_map_path: Union[str, Path],
) -> list[FMAAssessment]:
    """Load the clinical pre/post FMA assessment view over sourcedata."""

    sourcedata_root = Path(sourcedata_root)
    if not sourcedata_root.is_dir():
        raise FileNotFoundError(sourcedata_root)

    participant_rows = _read_tsv(
        participants_path, {"participant_id", "fma_pre", "fma_post"}
    )
    participants = {row["participant_id"]: row for row in participant_rows}
    session_map = _read_tsv(
        session_map_path,
        {
            "participant_id",
            "source_session_id",
            "assessment_id",
            "fma_session_id",
            "fma_score",
            "expected_mat_files",
        },
    )

    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in session_map:
        group_key = (
            row["participant_id"],
            row["assessment_id"],
            row["fma_session_id"],
            row["fma_score"],
        )
        grouped_rows.setdefault(group_key, []).append(row)

    assessments: list[FMAAssessment] = []
    for group_key in sorted(grouped_rows):
        rows = grouped_rows[group_key]
        participant_id, assessment_id, fma_session_id, score_text = group_key
        if participant_id not in participants:
            raise ValueError(f"Unknown participant in FMA map: {participant_id}")
        if assessment_id not in {"pre", "post"}:
            raise ValueError(
                f"Invalid assessment_id for {participant_id}: {assessment_id}"
            )

        fma_score = float(score_text)
        participant_score = float(participants[participant_id][f"fma_{assessment_id}"])
        if fma_score != participant_score:
            raise ValueError(
                f"FMA mismatch for {participant_id}/{assessment_id}: "
                f"map={fma_score}, participants={participant_score}"
            )

        source_sessions: list[str] = []
        mat_files: list[Path] = []
        for row in sorted(rows, key=lambda item: item["source_session_id"]):
            session_dir = (
                sourcedata_root
                / participant_id
                / row["source_session_id"]
                / "eeg"
            )
            files = sorted(session_dir.glob("*.mat"))
            expected = int(row["expected_mat_files"])
            if len(files) != expected:
                raise ValueError(
                    f"{participant_id}/{row['source_session_id']}: "
                    f"expected {expected} MAT files, found {len(files)}"
                )
            source_sessions.append(row["source_session_id"])
            mat_files.extend(files)

        assessments.append(
            FMAAssessment(
                participant_id=str(participant_id),
                assessment_id=str(assessment_id),
                fma_session_id=str(fma_session_id),
                fma_score=fma_score,
                source_session_ids=tuple(source_sessions),
                mat_files=tuple(mat_files),
            )
        )
    return assessments


def build_record_metadata(
    assessments: list[FMAAssessment],
) -> Dict[str, Dict[str, object]]:
    metadata: Dict[str, Dict[str, object]] = {}
    for assessment in assessments:
        for mat_file in assessment.mat_files:
            absolute_path = str(mat_file.resolve())
            source_session_id = mat_file.parent.parent.name
            if absolute_path in metadata:
                raise ValueError(f"FMA MAT file selected twice: {absolute_path}")
            metadata[absolute_path] = {
                "participant_id": assessment.participant_id,
                "assessment_id": assessment.assessment_id,
                "fma_session_id": assessment.fma_session_id,
                "fma_score": assessment.fma_score,
                "source_session_id": source_session_id,
            }
    return metadata


class FMADataset(BaseDataset):
    """Torcheeg dataset view for FMA regression sourced from ``sourcedata``."""

    def __init__(
        self,
        root_path: str,
        participants_path: str,
        session_map_path: str,
        io_path: Union[None, str] = None,
        duration: int = 1,
        num_channel: int = 11,
        sampling_rate: int = 128,
        l_freq: float = 8.0,
        h_freq: float = 48.0,
        online_transform: Union[None, Callable] = None,
        offline_transform: Union[None, Callable] = None,
        label_transform: Union[None, Callable] = None,
        before_trial: Union[None, Callable] = None,
        after_trial: Union[Callable, None] = None,
        after_session: Union[Callable, None] = None,
        after_subject: Union[Callable, None] = None,
        io_size: int = 1048576,
        io_mode: str = "lmdb",
        num_worker: int = 0,
        verbose: bool = True,
    ):
        if io_path is None:
            io_path = get_random_dir_path(dir_prefix="datasets")

        assessments = load_fma_assessments(
            sourcedata_root=root_path,
            participants_path=participants_path,
            session_map_path=session_map_path,
        )
        record_metadata = build_record_metadata(assessments)
        params = {
            "root_path": root_path,
            "participants_path": participants_path,
            "session_map_path": session_map_path,
            "record_metadata": record_metadata,
            "duration": duration,
            "num_channel": num_channel,
            "sampling_rate": sampling_rate,
            "l_freq": l_freq,
            "h_freq": h_freq,
            "online_transform": online_transform,
            "offline_transform": offline_transform,
            "label_transform": label_transform,
            "before_trial": before_trial,
            "after_trial": after_trial,
            "after_session": after_session,
            "after_subject": after_subject,
            "io_path": io_path,
            "io_size": io_size,
            "io_mode": io_mode,
            "num_worker": num_worker,
            "verbose": verbose,
        }
        super().__init__(**params)
        self.__dict__.update(params)

    @staticmethod
    def process_record(
        file: Any = None,
        duration: int = 1,
        sampling_rate: int = 128,
        l_freq: float = 8.0,
        h_freq: float = 48.0,
        num_channel: int = 11,
        before_trial: Union[None, Callable] = None,
        offline_transform: Union[None, Callable] = None,
        record_metadata: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs,
    ):
        mat_path = str(Path(file).resolve())
        if record_metadata is None or mat_path not in record_metadata:
            raise ValueError(f"Missing FMA metadata for {mat_path}")
        metadata = record_metadata[mat_path]

        mat = loadmat(mat_path, verify_compressed_data_integrity=False)
        run_samples = mat["EEGdata"].transpose(2, 0, 1)
        source_labels = mat["EEGdatalabel"][:, 0]
        ch_names = [
            ch[0].tolist()[0]
            for ch in mat["configuration_channel"][0]
            if ch[1].sum()
        ]
        if ch_names != DEFAULT_CHANNEL_LIST:
            raise ValueError(f"Incorrect channel list in {mat_path}: {ch_names}")

        info = mne.create_info(
            ch_names=[name.lower() for name in ch_names],
            sfreq=DEFAULT_SAMPLING_RATE,
            ch_types=["eeg"] * len(ch_names),
        )
        montage = mne.channels.make_standard_montage("standard_1020")
        montage.ch_names = [name.lower() for name in montage.ch_names]

        file_id = Path(mat_path).stem
        for run_id, (run_sample, source_label) in enumerate(
            zip(run_samples, source_labels)
        ):
            start = 9 * DEFAULT_SAMPLING_RATE
            end = 13 * DEFAULT_SAMPLING_RATE
            run_sample = run_sample[:, start:end]

            raw = mne.io.RawArray(run_sample, info, verbose=False)
            raw.set_montage(montage)
            raw = raw.filter(l_freq=l_freq, h_freq=h_freq)
            raw = raw.resample(sampling_rate)
            epochs = mne.make_fixed_length_epochs(
                raw,
                duration=duration,
                preload=True,
                verbose=False,
            )
            epochs, reject_stats = apply_autoreject(epochs, verbose=False)
            clips = normalize_clips(epochs.get_data())
            if before_trial:
                clips = before_trial(clips)

            for clip, window_id in zip(
                clips,
                reject_stats["kept_epoch_indices"],
            ):
                clip_id = (
                    f"{metadata['participant_id']}_"
                    f"{metadata['fma_session_id']}_{file_id}_"
                    f"trial-{run_id:03d}_window-{int(window_id):02d}"
                )
                record_info = {
                    "clip_id": clip_id,
                    "subject_id": metadata["participant_id"],
                    "session_id": metadata["fma_session_id"],
                    "assessment_id": metadata["assessment_id"],
                    "source_session_id": metadata["source_session_id"],
                    "file": file_id,
                    "trial_id": f"{file_id}_trial-{run_id:03d}",
                    "run_id": run_id,
                    "window_id": int(window_id),
                    "source_mi_label": int(source_label),
                    "fma_score": float(metadata["fma_score"]),
                    "label": float(metadata["fma_score"]) / FMA_SCALE,
                    "start_at": int(window_id) * duration * sampling_rate,
                    "end_at": (int(window_id) + 1) * duration * sampling_rate,
                    "autoreject": True,
                }
                if offline_transform:
                    clip = offline_transform(eeg=clip[:num_channel])["eeg"]
                yield {"eeg": clip, "key": clip_id, "info": record_info}

    def set_records(
        self,
        root_path: str,
        record_metadata: Optional[Dict[str, Dict[str, object]]] = None,
        **kwargs,
    ) -> list[str]:
        if record_metadata is None:
            raise ValueError("record_metadata is required for the FMA source view")
        records = sorted(record_metadata)
        if not records:
            raise RuntimeError("No FMA source records selected")
        return records

    def __getitem__(self, index: int) -> Tuple:
        info = self.read_info(index)
        eeg = self.read_eeg(str(info["_record_id"]), str(info["clip_id"]))
        label = info
        if self.online_transform:
            eeg = self.online_transform(eeg=eeg)["eeg"]
        if self.label_transform:
            label = self.label_transform(y=info)["y"]
        return eeg, label

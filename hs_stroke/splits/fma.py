"""FMA LOSO and LOSSO split implementations."""

from __future__ import annotations

import os
from typing import Generator, List, Optional, Tuple

import pandas as pd
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

from .base import _DatasetInfoView


class LeaveOneSubjectOut:
    """LOSO split where all data from one subject form the test fold."""

    def __init__(self, split_path: Optional[str] = None) -> None:
        self.split_path = split_path or get_random_dir_path(dir_prefix="model_selection")

    def _prepare_split_files(self, info: pd.DataFrame) -> None:
        os.makedirs(self.split_path, exist_ok=True)
        for file_name in os.listdir(self.split_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(self.split_path, file_name))

        for subject_id in sorted(info["subject_id"].astype(str).unique().tolist()):
            train_info = info[info["subject_id"] != subject_id]
            test_info = info[info["subject_id"] == subject_id]
            train_info.to_csv(os.path.join(self.split_path, f"train_subject_{subject_id}.csv"), index=False)
            test_info.to_csv(os.path.join(self.split_path, f"test_subject_{subject_id}.csv"), index=False)

    def split(
        self,
        dataset: BaseDataset,
        subject: Optional[str] = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, str], None, None]:
        if not os.path.exists(self.split_path):
            self._prepare_split_files(dataset.info)

        subjects: List[str] = sorted(dataset.info["subject_id"].astype(str).unique().tolist())
        if subject is not None:
            subjects = [subject]

        for subject_id in subjects:
            train_df = pd.read_csv(os.path.join(self.split_path, f"train_subject_{subject_id}.csv"))
            test_df = pd.read_csv(os.path.join(self.split_path, f"test_subject_{subject_id}.csv"))
            yield _DatasetInfoView(dataset, train_df), _DatasetInfoView(dataset, test_df), subject_id


class LeaveOneSubjectSessionOut:
    """LOSSO split where one ``(subject, session)`` pair is held out each fold."""

    def __init__(self, split_path: Optional[str] = None) -> None:
        self.split_path = split_path or get_random_dir_path(dir_prefix="model_selection")

    @staticmethod
    def _fold_key(subject_id: str, session_id: str) -> str:
        return f"{subject_id}__{session_id}"

    def _prepare_split_files(self, info: pd.DataFrame) -> None:
        missing = {"subject_id", "session_id"}.difference(info.columns)
        if missing:
            raise KeyError(f"Missing required columns for LOSSO split: {sorted(missing)}")

        os.makedirs(self.split_path, exist_ok=True)
        for file_name in os.listdir(self.split_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(self.split_path, file_name))

        unique_pairs = info[["subject_id", "session_id"]].drop_duplicates().sort_values(["subject_id", "session_id"])
        for _, pair in unique_pairs.iterrows():
            subject_id = str(pair["subject_id"])
            session_id = str(pair["session_id"])
            fold_key = self._fold_key(subject_id, session_id)
            test_mask = (info["subject_id"] == subject_id) & (info["session_id"] == session_id)
            info.loc[~test_mask].to_csv(
                os.path.join(self.split_path, f"train_subject_session_{fold_key}.csv"),
                index=False,
            )
            info.loc[test_mask].to_csv(
                os.path.join(self.split_path, f"test_subject_session_{fold_key}.csv"),
                index=False,
            )

    def split(
        self,
        dataset: BaseDataset,
        subject_session: Optional[str] = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, str], None, None]:
        if not os.path.exists(self.split_path):
            self._prepare_split_files(dataset.info)

        unique_pairs = dataset.info[["subject_id", "session_id"]].drop_duplicates().sort_values(["subject_id", "session_id"])
        fold_keys = [
            self._fold_key(str(row["subject_id"]), str(row["session_id"]))
            for _, row in unique_pairs.iterrows()
        ]
        if subject_session is not None:
            fold_keys = [subject_session]

        for fold_key in fold_keys:
            train_df = pd.read_csv(os.path.join(self.split_path, f"train_subject_session_{fold_key}.csv"))
            test_df = pd.read_csv(os.path.join(self.split_path, f"test_subject_session_{fold_key}.csv"))
            yield _DatasetInfoView(dataset, train_df), _DatasetInfoView(dataset, test_df), fold_key

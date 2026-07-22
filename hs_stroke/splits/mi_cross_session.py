"""MI cross-session split implementation."""

from __future__ import annotations

import logging
import os
import re
from typing import Generator, Optional, Tuple

import pandas as pd
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

from .base import _DatasetInfoView


log = logging.getLogger("hs_stroke")


class CrossSessionSplit:

    def __init__(
        self,
        split_path: Optional[str] = None,
    ) -> None:
        self.split_path = split_path or get_random_dir_path(dir_prefix="model_selection")

    @staticmethod
    def _session_to_int(session_id: str) -> int:
        matches = re.findall(r"(\d+)", str(session_id))
        if not matches:
            raise ValueError(f"Invalid session id: {session_id}")
        return int(matches[-1])

    def _subject_split(self, subject_info: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        session_nums = subject_info["session_id"].apply(self._session_to_int)
        unique_session_nums = sorted(set(session_nums.tolist()))

        if len(unique_session_nums) < 2:
            return subject_info.iloc[0:0], subject_info.iloc[0:0]
        held_out = set(unique_session_nums[::2])
        return (
            subject_info[~session_nums.isin(held_out)],
            subject_info[session_nums.isin(held_out)],
        )

    def _prepare_split_files(self, info: pd.DataFrame) -> None:
        os.makedirs(self.split_path, exist_ok=True)
        for file_name in os.listdir(self.split_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(self.split_path, file_name))

        for subject_id in sorted(info["subject_id"].astype(str).unique().tolist()):
            subject_info = info[info["subject_id"] == subject_id].copy()
            train_info, test_info = self._subject_split(subject_info)
            if len(train_info) == 0 or len(test_info) == 0:
                log.warning("Skipping subject %s because the split is empty.", subject_id)
                continue
            train_info.to_csv(os.path.join(self.split_path, f"train_subject_{subject_id}.csv"), index=False)
            test_info.to_csv(os.path.join(self.split_path, f"test_subject_{subject_id}.csv"), index=False)

    def split(
        self,
        dataset: BaseDataset,
        subject: Optional[str] = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, str], None, None]:
        if not os.path.exists(self.split_path):
            self._prepare_split_files(dataset.info)

        subjects = sorted(dataset.info["subject_id"].astype(str).unique().tolist())
        if subject is not None:
            subjects = [subject]

        for subject_id in subjects:
            train_path = os.path.join(self.split_path, f"train_subject_{subject_id}.csv")
            test_path = os.path.join(self.split_path, f"test_subject_{subject_id}.csv")
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                continue
            yield (
                _DatasetInfoView(dataset, pd.read_csv(train_path)),
                _DatasetInfoView(dataset, pd.read_csv(test_path)),
                subject_id,
            )

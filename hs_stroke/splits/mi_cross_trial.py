"""MI cross-trial split implementation."""

from __future__ import annotations

import logging
import os
import re
from typing import Generator, Optional

import pandas as pd
from sklearn.model_selection import KFold
from torcheeg.datasets import BaseDataset
from torcheeg.utils import get_random_dir_path

from .base import _DatasetInfoView


log = logging.getLogger("hs_stroke")


class CrossTrialSplit:

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int = 42,
        split_path: Optional[str] = None,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path or get_random_dir_path(
            dir_prefix="model_selection"
        )

    @staticmethod
    def _safe_session(session_id: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session_id))

    def _split_file(
        self,
        part: str,
        subject_id: str,
        session_id: str,
        fold_id: int,
    ) -> str:
        return os.path.join(
            self.split_path,
            f"{part}_subject_{subject_id}_session_{self._safe_session(session_id)}"
            f"_fold_{fold_id}.csv",
        )

    def _eligible_pairs(self, info: pd.DataFrame) -> list[tuple[str, str]]:
        pairs = []
        normalized = info.assign(
            subject_id=info["subject_id"].astype(str),
            session_id=info["session_id"].astype(str),
            trial_id=info["trial_id"].astype(str),
        )
        for pair, session_info in normalized.groupby(
            ["subject_id", "session_id"],
            sort=True,
        ):
            if session_info["trial_id"].nunique() >= self.n_splits:
                pairs.append(pair)
        return pairs

    def _prepare_split_files(self, info: pd.DataFrame) -> None:
        os.makedirs(self.split_path, exist_ok=True)
        for file_name in os.listdir(self.split_path):
            if file_name.endswith(".csv"):
                os.remove(os.path.join(self.split_path, file_name))

        normalized = info.assign(
            subject_id=info["subject_id"].astype(str),
            session_id=info["session_id"].astype(str),
            trial_id=info["trial_id"].astype(str),
        )
        for subject_id, session_id in self._eligible_pairs(normalized):
            session_info = normalized[
                (normalized["subject_id"] == subject_id)
                & (normalized["session_id"] == session_id)
            ].copy()
            trial_ids = sorted(session_info["trial_id"].unique().tolist())
            splitter = KFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state if self.shuffle else None,
            )
            for fold_id, (train_indices, test_indices) in enumerate(
                splitter.split(trial_ids)
            ):
                train_trials = {trial_ids[index] for index in train_indices}
                test_trials = {trial_ids[index] for index in test_indices}
                if train_trials.intersection(test_trials):
                    raise RuntimeError("A complete trial crossed the fold boundary")
                session_info[session_info["trial_id"].isin(train_trials)].to_csv(
                    self._split_file("train", subject_id, session_id, fold_id),
                    index=False,
                )
                session_info[session_info["trial_id"].isin(test_trials)].to_csv(
                    self._split_file("test", subject_id, session_id, fold_id),
                    index=False,
                )
            log.info(
                "%s %s: %d trials -> %d folds",
                subject_id,
                session_id,
                len(trial_ids),
                self.n_splits,
            )

    def _split_files_complete(self, info: pd.DataFrame) -> bool:
        if not os.path.isdir(self.split_path):
            return False
        return all(
            os.path.isfile(self._split_file(part, subject, session, fold))
            for subject, session in self._eligible_pairs(info)
            for fold in range(self.n_splits)
            for part in ("train", "test")
        )

    def split(
        self,
        dataset: BaseDataset,
        subject: Optional[str] = None,
    ) -> Generator[tuple[BaseDataset, BaseDataset, str, str, int], None, None]:
        if not self._split_files_complete(dataset.info):
            self._prepare_split_files(dataset.info)

        known_subjects = set(dataset.info["subject_id"].astype(str))
        if subject is not None and str(subject) not in known_subjects:
            raise ValueError(f"Unknown subject: {subject}")

        for subject_id, session_id in self._eligible_pairs(dataset.info):
            if subject is not None and subject_id != str(subject):
                continue
            for fold_id in range(self.n_splits):
                train_info = pd.read_csv(
                    self._split_file("train", subject_id, session_id, fold_id)
                )
                test_info = pd.read_csv(
                    self._split_file("test", subject_id, session_id, fold_id)
                )
                train_trials = set(train_info["trial_id"].astype(str))
                test_trials = set(test_info["trial_id"].astype(str))
                if train_trials.intersection(test_trials):
                    raise RuntimeError(
                        f"Trial leakage: {subject_id}/{session_id}/fold-{fold_id + 1}"
                    )
                yield (
                    _DatasetInfoView(dataset, train_info),
                    _DatasetInfoView(dataset, test_info),
                    subject_id,
                    session_id,
                    fold_id,
                )

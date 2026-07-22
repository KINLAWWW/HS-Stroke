"""Dataset views shared by split implementations."""

from __future__ import annotations

import pandas as pd
from torcheeg.datasets import BaseDataset


class _DatasetInfoView:
    """Replace only ``info`` while reusing the underlying dataset cache."""

    def __init__(self, dataset: BaseDataset, info: pd.DataFrame) -> None:
        self.dataset = dataset
        self.info = info.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.info)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __copy__(self):
        return _DatasetInfoView(self.dataset, self.info.copy())

    def __getitem__(self, index: int):
        info = self.info.iloc[index].to_dict()
        eeg = self.dataset.read_eeg(str(info["_record_id"]), str(info["clip_id"]))

        signal = eeg
        label = info
        if getattr(self.dataset, "online_transform", None):
            signal = self.dataset.online_transform(eeg=eeg)["eeg"]
        if getattr(self.dataset, "label_transform", None):
            label = self.dataset.label_transform(y=info)["y"]
        return signal, label

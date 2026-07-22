"""EEGNet classifier."""

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: int = 1, **kwargs):
        self.max_norm = max_norm
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(self,
                 chunk_size: int = 128,
                 num_electrodes: int = 11,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 num_classes: int = 2,
                 kernel_1: int = 64,
                 kernel_2: int = 16,
                 dropout: float = 0.25):
        super().__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_1), stride=1, padding=(0, kernel_1 // 2), bias=False),
            nn.BatchNorm2d(F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(F1,
                                 F1 * D,
                                 (num_electrodes, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=F1,
                                 bias=False),
            nn.BatchNorm2d(F1 * D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D,
                      F1 * D,
                      (1, kernel_2),
                      stride=1,
                      padding=(0, kernel_2 // 2),
                      bias=False,
                      groups=F1 * D),
            nn.Conv2d(F1 * D, F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout)
        )

        self.lin = nn.Linear(self.feature_dim(), num_classes, bias=False)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)
        return self.F2 * mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(start_dim=1)
        x = self.lin(x)
        return x

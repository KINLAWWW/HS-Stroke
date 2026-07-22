"""ATCNet regressor."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATCNet_R(nn.Module):
    """ATCNet regression model."""

    def __init__(
        self,
        in_channels: int = 1,
        output_dim: int = 1,
        num_windows: int = 3,
        num_electrodes: int = 22,
        conv_pool_size: int = 7,
        F1: int = 16,
        D: int = 2,
        tcn_kernel_size: int = 4,
        tcn_depth: int = 2,
        chunk_size: int = 1125,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.output_dim = output_dim
        self.num_windows = num_windows
        self.num_electrodes = num_electrodes
        self.pool_size = conv_pool_size
        self.F1 = F1
        self.D = D
        self.tcn_kernel_size = tcn_kernel_size
        self.tcn_depth = tcn_depth
        self.chunk_size = chunk_size

        F2 = F1 * D

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, F1, (1, int(chunk_size / 2 + 1)), stride=1, padding="same", bias=False),
            nn.BatchNorm2d(F1, affine=False),
            nn.Conv2d(F1, F2, (num_electrodes, 1), padding=0, groups=F1),
            nn.BatchNorm2d(F2, affine=False),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout2d(0.1),

            nn.Conv2d(F2, F2, (1, 16), padding="same", bias=False),
            nn.BatchNorm2d(F2, affine=False),
            nn.ELU(),
            nn.AvgPool2d((1, self.pool_size)),
            nn.Dropout2d(0.1),
        )

        self._build_model()

    def _build_model(self):
        with torch.no_grad():
            x = torch.zeros(2, self.in_channels, self.num_electrodes, self.chunk_size)
            x = self.conv_block(x)
            x = x[:, :, -1, :]
            x = x.permute(0, 2, 1)

            self._seq_len, self._embed_dim = x.shape[1:]
            self.win_len = self._seq_len - self.num_windows + 1

            for i in range(self.num_windows):
                st = i
                end = x.shape[1] - self.num_windows + i + 1
                x2 = x[:, st:end, :]

                self._add_msa(i)
                attn_out = self.get_submodule(f"msa{i}")(x2, x2, x2)[0]
                self._add_msa_drop(i)
                attn_out = self.get_submodule(f"msa_drop{i}")(attn_out)
                x2 = x2 + attn_out

                for j in range(self.tcn_depth):
                    idx = (i + 1) * j
                    self._add_tcn(idx, x2.shape[1])
                    out = self.get_submodule(f"tcn{idx}")(x2)

                    if x2.shape[1] != out.shape[1]:
                        self._add_recov(i)
                        x2 = self.get_submodule(f"re{i}")(x2)

                    x2 = x2 + out
                    x2 = nn.ELU()(x2)

                x2 = x2[:, -1, :]
                self._dense_dim = x2.shape[-1]
                self._add_dense(i)

    def _add_msa(self, index: int):
        self.add_module(
            f"msa{index}",
            nn.MultiheadAttention(embed_dim=self._embed_dim, num_heads=2, batch_first=True),
        )

    def _add_msa_drop(self, index: int):
        self.add_module(f"msa_drop{index}", nn.Dropout(0.3))

    def _add_tcn(self, index: int, seq_len: int):
        self.add_module(
            f"tcn{index}",
            nn.Sequential(
                nn.Conv1d(seq_len, 32, self.tcn_kernel_size, padding="same"),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),

                nn.Conv1d(32, 32, self.tcn_kernel_size, padding="same"),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.3),
            ),
        )

    def _add_recov(self, index: int):
        self.add_module(f"re{index}", nn.Conv1d(self.win_len, 32, 4, padding="same"))

    def _add_dense(self, index: int):
        self.add_module(f"dense{index}", nn.Linear(self._dense_dim, self.output_dim))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv_block(x)
        x = x[:, :, -1, :]
        x = x.permute(0, 2, 1)

        for i in range(self.num_windows):
            st = i
            end = x.shape[1] - self.num_windows + i + 1
            x2 = x[:, st:end, :]

            attn = self.get_submodule(f"msa{i}")(x2, x2, x2)[0]
            attn = self.get_submodule(f"msa_drop{i}")(attn)
            x2 = x2 + attn

            for j in range(self.tcn_depth):
                idx = (i + 1) * j
                out = self.get_submodule(f"tcn{idx}")(x2)

                if x2.shape[1] != out.shape[1]:
                    x2 = self.get_submodule(f"re{i}")(x2)

                x2 = x2 + out
                x2 = nn.ELU()(x2)

            x2 = x2[:, -1, :]
            x2 = self.get_submodule(f"dense{i}")(x2)

            if i == 0:
                sw_concat = x2
            else:
                sw_concat = sw_concat + x2

        x = sw_concat / self.num_windows
        return x

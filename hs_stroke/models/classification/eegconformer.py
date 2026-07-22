"""EEGConformer classifier."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self,
                 num_electrodes: int,
                 hid_channels: int = 40,
                 dropout: float = 0.5):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, hid_channels, (1, 25), (1, 1)),
            nn.Conv2d(hid_channels, hid_channels, (num_electrodes, 1), (1, 1)),
            nn.BatchNorm2d(hid_channels),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, (1, 1), stride=(1, 1)),
            Rearrange('b e h w -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hid_channels: int, heads: int, dropout: float):
        super().__init__()
        self.hid_channels = hid_channels
        self.heads = heads
        self.keys = nn.Linear(hid_channels, hid_channels)
        self.queries = nn.Linear(hid_channels, hid_channels)
        self.values = nn.Linear(hid_channels, hid_channels)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(hid_channels, hid_channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        queries = rearrange(self.queries(x), 'b n (h d) -> b h n d', h=self.heads)
        keys = rearrange(self.keys(x), 'b n (h d) -> b h n d', h=self.heads)
        values = rearrange(self.values(x), 'b n (h d) -> b h n d', h=self.heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.hid_channels ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav', att, values)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, hid_channels: int, expansion: int = 4, dropout: float = 0.):
        super().__init__(
            nn.Linear(hid_channels, expansion * hid_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * hid_channels, hid_channels),
        )


class GELU(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, hid_channels: int, heads: int, dropout: float,
                 forward_expansion: int, forward_dropout: float):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    MultiHeadAttention(hid_channels, heads, dropout),
                    nn.Dropout(dropout)
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(hid_channels),
                    FeedForwardBlock(hid_channels,
                                     expansion=forward_expansion,
                                     dropout=forward_dropout),
                    nn.Dropout(dropout)
                )
            )
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self,
                 depth: int,
                 hid_channels: int,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5):
        super().__init__(*[TransformerEncoderBlock(hid_channels=hid_channels,
                                                   heads=heads,
                                                   dropout=dropout,
                                                   forward_expansion=forward_expansion,
                                                   forward_dropout=forward_dropout)
                           for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 num_classes: int,
                 hid_channels: int = 32,
                 dropout: float = 0.5):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hid_channels * 8),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels * 8, hid_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hid_channels, num_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


class EEGConformer(nn.Module):
    def __init__(self,
                 num_electrodes: int = 11,
                 sampling_rate: int = 128,
                 embed_dropout: float = 0.5,
                 hid_channels: int = 40,
                 depth: int = 6,
                 heads: int = 10,
                 dropout: float = 0.5,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0.5,
                 cls_channels: int = 32,
                 cls_dropout: float = 0.5,
                 num_classes: int = 2):
        super().__init__()
        self.num_electrodes = num_electrodes
        self.sampling_rate = sampling_rate

        self.embd = PatchEmbedding(num_electrodes, hid_channels, embed_dropout)
        self.encoder = TransformerEncoder(depth,
                                          hid_channels,
                                          heads=heads,
                                          dropout=dropout,
                                          forward_expansion=forward_expansion,
                                          forward_dropout=forward_dropout)
        self.cls = ClassificationHead(in_channels=self.feature_dim(),
                                      num_classes=num_classes,
                                      hid_channels=cls_channels,
                                      dropout=cls_dropout)

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.sampling_rate)
            mock_eeg = self.embd(mock_eeg)
            mock_eeg = self.encoder(mock_eeg)
            mock_eeg = mock_eeg.flatten(start_dim=1)
            return mock_eeg.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embd(x)
        x = self.encoder(x)
        x = self.cls(x)
        return x

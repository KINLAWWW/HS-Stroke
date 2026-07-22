"""IFNet regressor."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class Conv(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        self.activation = activation
        if bn:
            try:
                self.conv.bias = None
            except Exception:
                pass
        self.bn = bn

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFre(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_list):
        out = sum(x_list)
        out = F.gelu(out)
        return out


class Stem(nn.Module):
    """IFNet stem block."""
    def __init__(self, in_planes, out_planes=64, kernel_size=63,
                 patch_size=125, radix=2, dropout=0.5):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.mid_planes = out_planes * radix
        self.kernel_size = kernel_size
        self.radix = radix

        self.sconv = Conv(
            nn.Conv1d(self.in_planes, self.mid_planes, kernel_size=1,
                      bias=False, groups=radix),
            bn=nn.BatchNorm1d(self.mid_planes),
            activation=None
        )

        self.tconv = nn.ModuleList()
        kk = kernel_size
        for _ in range(self.radix):
            self.tconv.append(
                Conv(
                    nn.Conv1d(self.out_planes, self.out_planes, kernel_size=kk,
                              stride=1, padding=kk // 2, groups=self.out_planes,
                              bias=False),
                    bn=nn.BatchNorm1d(self.out_planes),
                    activation=None
                )
            )
            kk = max(1, kk // 2)

        self.interFre = InterFre()
        self.downSampling = nn.AvgPool1d(kernel_size=patch_size,
                                         stride=patch_size)
        self.dp = nn.Dropout(dropout)

    def forward(self, x):
        out = self.sconv(x)
        out_splits = torch.split(out, self.out_planes, dim=1)
        out_conv = [m(s) for s, m in zip(out_splits, self.tconv)]
        out = self.interFre(out_conv)
        out = self.downSampling(out)
        out = self.dp(out)
        return out


class IFNet_R(nn.Module):
    """IFNet regression model."""
    def __init__(self, in_planes=11, out_planes=64, kernel_size=63,
                 radix=2, patch_size=125, dropout_fc=0.5):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.stem = Stem(
            self.in_planes, out_planes, kernel_size,
            patch_size=patch_size, radix=radix, dropout=0.5
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_fc),
            nn.Linear(self.out_planes, 1)
        )

        self.apply(self.init_parms)

    def init_parms(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if getattr(m, 'weight', None) is not None:
                nn.init.constant_(m.weight, 1.0)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.01)
            if getattr(m, 'bias', None) is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        out = self.stem(x)
        out = self.global_pool(out)
        features = out.squeeze(-1)
        out = self.fc(features)
        if return_features:
            return out, features
        return out

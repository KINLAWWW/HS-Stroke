"""Model package."""

from .classification import ATCNet, ArjunViT, EEGConformer, EEGNet, IFNet
from .regression import ATCNet_R, ArjunViT_R, IFNet_R

__all__ = [
    "ATCNet",
    "ATCNet_R",
    "ArjunViT",
    "ArjunViT_R",
    "EEGConformer",
    "EEGNet",
    "IFNet",
    "IFNet_R",
]

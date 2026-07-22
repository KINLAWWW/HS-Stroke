"""MI classification models."""

from .arjunvit import ArjunViT
from .atcnet import ATCNet
from .eegconformer import EEGConformer
from .eegnet import EEGNet
from .ifnet import IFNet

__all__ = ["ATCNet", "ArjunViT", "EEGConformer", "EEGNet", "IFNet"]

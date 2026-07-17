"""Quaternion autoencoders: linear and convolutional variants."""

from .linear import QuaternionLinearAE
from .convolutional import HyperkahlerConvAE

__all__ = ["QuaternionLinearAE", "HyperkahlerConvAE"]

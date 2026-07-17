"""Unified Hyperkähler AI framework combining quaternion autoencoders and transformers."""

__version__ = "0.1.0"
__author__ = "Subunits"

from .core import QuaternionOps, GeometryOps
from .fusion import ModelCombo

__all__ = [
    "QuaternionOps",
    "GeometryOps", 
    "ModelCombo",
]

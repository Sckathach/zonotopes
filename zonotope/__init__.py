import torch as t

from zonotope.utils import dual_norm
from zonotope.zonotope import Zonotope

DEVICE: str = "cuda" if t.cuda.is_available() else "cpu"

__all__ = [
    "Zonotope",
    "dual_norm",
]

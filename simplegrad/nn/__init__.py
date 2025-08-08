from .Linear import Linear
from .weight_init import xavier_init
from .Module import Module
from .Parameter import Parameter
from .Sequential import Sequential
from . import functional

try:
    from .MNISTLoader import MNISTLoader
except Exception:  # optional dependency (datasets)
    MNISTLoader = None

__all__ = [
    "Linear",
    "xavier_init",
    "Module",
    "Parameter",
    "Sequential",
    "functional",
    "MNISTLoader",
]

from typing import Callable
import numpy as np


def get_numeric(f: Callable, x: np.ndarray) -> np.ndarray:
    h = 0.0000000001
    return (f(x + h) - f(x)) / h

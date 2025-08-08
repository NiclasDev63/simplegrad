from typing import Callable
import numpy as np


def get_numeric(f: Callable, *inputs: np.ndarray) -> np.ndarray:
    h = 0.0000000001
    next_inputs = [x + h for x in inputs]
    return (f(*next_inputs) - f(*inputs)) / h

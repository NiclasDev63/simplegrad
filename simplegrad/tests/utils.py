from typing import Callable
import numpy as np


def get_numeric(f: Callable, *inputs: np.ndarray, direction: int = 0) -> np.ndarray:
    """
    Numerically estimates the partial derivative of a function with respect to one input.

    Args:
        f (Callable): The function to differentiate. Should accept the same number of inputs as provided.
        *inputs (np.ndarray): The input arrays to the function.
        direction (int): The index of the input to perturb for the derivative. (default 0)

    Returns:
        np.ndarray: The estimated partial derivative of f with respect to the input at 'direction'.
    """
    h = 0.0000000001
    next_inputs = [x + h if idx == direction else x for idx, x in enumerate(inputs)]
    return (f(*next_inputs) - f(*inputs)) / h

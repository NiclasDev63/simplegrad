import numpy as np
from typing import Optional, Tuple


def unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]):
    """
    Sum grad over axes that were broadcasted to match the original shape.
    """
    # If shape is (), treat as scalar
    if shape == ():
        return np.sum(grad)
    # Add leading 1s to shape if needed
    while len(shape) < grad.ndim:
        shape = (1,) + shape
    # Sum over axes where original shape is 1 but grad shape is >1
    axis = tuple(
        i for i, (s, g) in enumerate(zip(shape, grad.shape)) if s == 1 and g > 1
    )
    if axis:
        grad = np.sum(grad, axis=axis, keepdims=True)

    grad = grad.reshape(shape)
    return grad


def handle_axis(axis: Optional[int | tuple[int, int] | np.ndarray], ndim: int):
    if isinstance(axis, np.ndarray):
        if axis.dtype == object:
            axis = None
        else:
            axis = tuple(axis.tolist())

    if axis is None or np.isnan(axis):
        return None

    if np.isscalar(axis):
        axis = (axis,)

    # Handle negative axes
    axis = tuple([int(a) if a >= 0 else int(a + ndim) for a in axis])
    return axis
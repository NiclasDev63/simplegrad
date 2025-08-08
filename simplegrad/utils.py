import numpy as np
from typing import Optional, Tuple, Union


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


def handle_axis(axis: Optional[Union[int, tuple[int, int]]], ndim: int):

    if axis is None:
        return None
    if np.isscalar(axis):
        if np.isnan(axis):
            return None
    elif isinstance(axis, tuple):
        # If any element in the tuple is nan, treat as None
        if any(np.isnan(a) for a in axis):
            return None

    if np.isscalar(axis):
        axis = (axis,)

    # Handle negative axes
    axis = tuple([int(a) if a >= 0 else int(a + ndim) for a in axis])
    return axis

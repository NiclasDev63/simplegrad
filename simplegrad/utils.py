import numpy as np
from typing import Any, Optional, Tuple, Union

from simplegrad import Tensor
from simplegrad.Ops import Function


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


def assert_unmodified(*inputs: Tuple[int, Tensor]):
    """Assert saved tensors (and their view bases) weren't modified in-place."""

    visited: set[int] = set()

    def check(x: object) -> bool:
        if not Function.is_pair(x):
            return True
        saved_version, t = x  # type: ignore[misc]
        if t._version != saved_version:
            return False
        tid = id(t)
        if tid in visited:
            return True
        visited.add(tid)
        ctx = t._ctx
        if ctx and (
            getattr(ctx, "is_view", False)
            or ctx.__class__.__name__ in {"Index", "Reshape"}
        ):
            return all(check(s) for s in getattr(ctx, "saved_tensors", []))
        return True

    assert all(
        check(inp) for inp in inputs
    ), "one of the variables needed for gradient computation has been modified by an inplace operation"


def _assert_allowed_backward_call(x: Tensor, other: Any):
    if Function.is_pair(other):
        assert_unmodified(other)
    # only check other value if we have to compute gradients w.r.t. other.
    # otherwise it doesnt matter if the current value i.e. x was modified
    if Function.is_pair(x) and Function.is_pair(other):
        assert_unmodified(x)

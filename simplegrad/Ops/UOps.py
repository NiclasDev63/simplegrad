from typing import Any, Optional, Tuple, Union

from simplegrad import Tensor
from .Function import Function, ValueLike
import numpy as np
from simplegrad.utils import handle_axis, _assert_allowed_backward_call


class Sum(Function):
    @staticmethod
    def forward(
        ctx: "Sum",
        x: Tensor,
        axis: Optional[int | tuple[int, int]] = None,
    ) -> np.ndarray:
        x, axis = Function._unwrap_args(x, axis)

        axis = handle_axis(axis, ndim=x.ndim)
        ctx.save_for_backward(x, axis)

        return np.sum(x, axis=axis)

    @staticmethod
    def backward(ctx: "Sum", grad_output: np.ndarray) -> np.ndarray:
        x, axis = ctx.saved_tensors

        if axis is None:
            # Sum over all elements - gradient should be broadcast to original shape
            return np.full_like(x, grad_output)
        else:
            # Sum over specific axes - gradient needs to be expanded
            # Create shape for broadcasting
            shape = list(x.shape)
            for ax in axis:
                shape[ax] = 1
            return np.broadcast_to(grad_output.reshape(shape), x.shape)


class Reshape(Function):
    @staticmethod
    def forward(ctx: "Reshape", x: Tensor, shape: Tuple[int, ...]) -> np.ndarray:
        x = Function._unwrap_args(x)
        ctx.save_for_backward(x)

        shape = [int(s) for s in shape]
        return x.reshape(shape)

    @staticmethod
    def backward(ctx: "Reshape", grad_output: np.ndarray) -> np.ndarray:
        x = ctx.saved_tensors[0]
        return grad_output.reshape(x.shape)


class Index(Function):
    @staticmethod
    def forward(ctx: "Index", x: Tensor, index: Any) -> np.ndarray:
        ctx.save_for_backward(x, index)
        x, index = Function._unwrap_args(x, index)
        return x[index]

    @staticmethod
    def backward(ctx: "Index", grad_output: np.ndarray) -> np.ndarray:
        x, index = ctx.saved_tensors
        _assert_allowed_backward_call(x, index)
        x, index = Function._unwrap_args(x, index)
        grad = np.zeros_like(x)
        grad[index] = grad_output
        return grad


class CopySlices(Function):
    _in_place = True

    @staticmethod
    def forward(
        ctx: "CopySlices", x: Tensor, index: Any, value: ValueLike
    ) -> np.ndarray:
        x, index, value = Function._unwrap_args(x, index, value)
        ctx.save_for_backward(index)
        x[index] = value
        return x

    @staticmethod
    def backward(ctx: "CopySlices", grad_output: np.ndarray):
        index = ctx.saved_tensors[0]

        # grad w.r.t. base (old_self): identity except written slice → zeroed
        grad_base = grad_output.copy()
        grad_base[index] = 0

        # If RHS is a Tensor parent, also return its grad (the written slice)
        if len(ctx.parents) > 1:
            grad_value = grad_output[index]
            return grad_base, grad_value

        return grad_base


class Relu(Function):
    @staticmethod
    def _forward(x: np.ndarray) -> np.ndarray:
        return x.clip(min=0)

    @staticmethod
    def forward(ctx: "Relu", x: Tensor) -> np.ndarray:
        x = Function._unwrap_args(x)
        ctx.save_for_backward(x)

        return Relu._forward(x)

    @staticmethod
    def backward(ctx: "Relu", grad_output: np.ndarray) -> np.ndarray:
        x = ctx.saved_tensors[0]

        grad = grad_output.copy()
        grad[x < 0] = 0
        return grad


class Sigmoid(Function):
    @staticmethod
    def _forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def forward(ctx: "Sigmoid", x: Tensor) -> np.ndarray:
        x = Function._unwrap_args(x)
        ctx.save_for_backward(x)

        return Sigmoid._forward(x)

    @staticmethod
    def backward(ctx: "Sigmoid", grad_output: np.ndarray) -> np.ndarray:
        x = ctx.saved_tensors[0]

        sig = Sigmoid._forward(x)
        grad = sig * (1 - sig)
        return grad * grad_output


class Exp(Function):
    @staticmethod
    def forward(ctx: "Exp", x: Tensor) -> np.ndarray:
        x = Function._unwrap_args(x)

        x_exp = np.exp(x)
        ctx.save_for_backward(x_exp)
        return x_exp

    @staticmethod
    def backward(ctx: "Exp", grad_output: np.ndarray) -> np.ndarray:
        x_exp = ctx.saved_tensors[0]
        return x_exp * grad_output


class Log(Function):
    @staticmethod
    def forward(ctx: "Log", x: Tensor) -> np.ndarray:
        x = Function._unwrap_args(x)
        x = x + 1e-8
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx: "Log", grad_output: np.ndarray) -> np.ndarray:
        x = ctx.saved_tensors[0]
        return grad_output / x


class Neg(Function):
    @staticmethod
    def forward(ctx: "Neg", x: Tensor) -> np.ndarray:
        x = Function._unwrap_args(x)
        return -x

    @staticmethod
    def backward(ctx: "Neg", grad_output: np.ndarray) -> np.ndarray:
        return -grad_output


class Softmax(Function):
    @staticmethod
    def _forward(
        x: np.ndarray,
        axis: Union[int, tuple[int, int]],
        temperature: float = 1,
    ) -> np.ndarray:
        x = x / temperature

        # Subtract max for numerical stability
        x_max = np.max(x, axis=axis, keepdims=True)
        x_stable = x - x_max
        exp_x = np.exp(x_stable)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        softmax_output = exp_x / (sum_exp + 1e-8)
        return softmax_output

    @staticmethod
    def forward(
        ctx: "Softmax",
        x: Tensor,
        axis: Union[int, tuple[int, int]],
        temperature: float,
    ) -> np.ndarray:
        x = Function._unwrap_args(x)
        softmax_output = Softmax._forward(x, axis=axis, temperature=temperature)
        ctx.save_for_backward(softmax_output, axis)
        return softmax_output

    @staticmethod
    def backward(ctx: "Softmax", grad_output: np.ndarray) -> np.ndarray:
        softmax_output, axis = ctx.saved_tensors

        grad_sum = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
        grad = softmax_output * (grad_output - grad_sum)

        return grad

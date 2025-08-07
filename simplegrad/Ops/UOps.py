from tkinter.constants import X
from typing import Optional, Tuple
from .Function import Function
import numpy as np
from utils import handle_axis


class Sum(Function):
    @staticmethod
    def forward(
        ctx: "Sum",
        x: np.ndarray,
        axis: Optional[int | tuple[int, int] | np.ndarray] = None,
    ) -> np.ndarray:
        axis = handle_axis(axis, ndim=x.ndim)
        ctx.save_for_backward(x, axis)
        return np.sum(x, axis=axis)

    @staticmethod
    def backward(ctx: "Sum", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    def forward(ctx: "Reshape", x: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        ctx.save_for_backward(x)
        shape = [int(s) for s in shape]
        return x.reshape(shape)

    @staticmethod
    def backward(
        ctx: "Reshape", grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = ctx.saved_tensors[0]
        return grad_output.reshape(x.shape)


class Relu(Function):
    @staticmethod
    def forward(ctx: "Relu", x: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x)
        return x.clip(min=0)

    @staticmethod
    def backward(ctx: "Relu", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = ctx.saved_tensors[0]
        grad = grad_output.copy()
        grad[x < 0] = 0
        return grad


class Sigmoid(Function):
    @staticmethod
    def _forward(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def forward(ctx: "Sigmoid", x: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x)
        return Sigmoid._forward(x)

    @staticmethod
    def backward(
        ctx: "Sigmoid", grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        x = ctx.saved_tensors[0]
        sig = Sigmoid._forward(x)
        grad = sig * (1 - sig)
        return grad * grad_output


class Exp(Function):
    @staticmethod
    def forward(ctx: "Exp", x: np.ndarray) -> np.ndarray:
        x_exp = np.exp(x)
        ctx.save_for_backward(x_exp)
        return x_exp

    @staticmethod
    def backward(ctx: "Exp", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_exp = ctx.saved_tensors[0]
        return x_exp * grad_output


class Log(Function):
    @staticmethod
    def forward(ctx: "Log", x: np.ndarray) -> np.ndarray:
        x = x + 1e-8
        ctx.save_for_backward(x)
        return np.log(x)

    @staticmethod
    def backward(ctx: "Log", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = ctx.saved_tensors[0]
        return grad_output / x


class Neg(Function):
    @staticmethod
    def forward(ctx: "Neg", x: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x)
        return -x

    @staticmethod
    def backward(ctx: "Neg", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return -grad_output


# TODO: add optional temperature
class Softmax(Function):
    @staticmethod
    def forward(ctx: "Softmax", x: np.ndarray, axis: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability
        axis = int(axis[0])
        x_max = np.max(x, axis=axis, keepdims=True)
        x_stable = x - x_max
        exp_x = np.exp(x_stable)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        softmax_output = exp_x / (sum_exp + 1e-8)

        ctx.save_for_backward(softmax_output, axis)
        return softmax_output

    @staticmethod
    def backward(
        ctx: "Softmax", grad_output: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        softmax_output, axis = ctx.saved_tensors

        # Gradient for softmax: softmax * (grad_output - sum(grad_output * softmax, axis=axis))
        grad_sum = np.sum(grad_output * softmax_output, axis=axis, keepdims=True)
        grad = softmax_output * (grad_output - grad_sum)

        return grad

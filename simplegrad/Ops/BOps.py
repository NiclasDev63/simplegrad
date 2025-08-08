from typing import Tuple
import numpy as np

from simplegrad import Tensor
from simplegrad.utils import _assert_allowed_backward_call
from .Function import Function, ValueLike


class Add(Function):
    @staticmethod
    def forward(ctx: "Add", x: Tensor, y: ValueLike) -> np.ndarray:
        x, y = Function._unwrap_args(x, y)
        return x + y

    @staticmethod
    def backward(ctx: "Add", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: "Sub", x: Tensor, y: ValueLike) -> np.ndarray:
        x, y = Function._unwrap_args(x, y)
        return x - y

    @staticmethod
    def backward(ctx: "Sub", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: "Mul", x: Tensor, y: ValueLike) -> np.ndarray:
        ctx.save_for_backward(x, y)
        x, y = Function._unwrap_args(x, y)
        return x * y

    @staticmethod
    def backward(ctx: "Mul", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_saved, y_saved = ctx.saved_tensors
        _assert_allowed_backward_call(x_saved, y_saved)
        x_in, y_in = Function._unwrap_args(x_saved, y_saved)

        x_grad = grad_output * y_in
        y_grad = grad_output * x_in
        return x_grad, y_grad


class Div(Function):
    @staticmethod
    def forward(ctx: "Div", x: Tensor, y: ValueLike) -> np.ndarray:
        from simplegrad import Tensor

        if isinstance(y, Tensor):
            y.item = y.item + 1e-8
        else:
            y = y + 1e-8
        ctx.save_for_backward(x, y)
        x, y = Function._unwrap_args(x, y)
        return x / y

    @staticmethod
    def backward(ctx: "Div", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_saved, y_saved = ctx.saved_tensors
        _assert_allowed_backward_call(x_saved, y_saved)
        x_in, y_in = Function._unwrap_args(x_saved, y_saved)

        x_grad = (1 / y_in) * grad_output
        y_grad = -x_in * (1 / (y_in**2)) * grad_output

        return x_grad, y_grad


class Pow(Function):
    @staticmethod
    def forward(ctx: "Pow", base: Tensor, exponent: ValueLike) -> np.ndarray:
        ctx.save_for_backward(base, exponent)
        base, exponent = Function._unwrap_args(base, exponent)
        return np.pow(base, exponent)

    @staticmethod
    def backward(ctx: "Pow", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base, exponent = ctx.saved_tensors
        _assert_allowed_backward_call(base, exponent)
        base, exponent = Function._unwrap_args(base, exponent)

        exponent = np.array(exponent)

        result_shape = np.broadcast(base, exponent).shape
        grad_output_b = np.broadcast_to(grad_output, result_shape)
        base_b = np.broadcast_to(base, result_shape)
        exponent_b = np.broadcast_to(exponent, result_shape)

        base_grad = exponent_b * np.power(base_b, exponent_b - 1) * grad_output_b

        safe_mask = base_b > 0
        local_all = np.zeros_like(grad_output_b)
        # compute only on safe region to avoid log of non-positive
        local_all[safe_mask] = np.power(
            base_b[safe_mask], exponent_b[safe_mask]
        ) * np.log(base_b[safe_mask])

        if exponent.size == 1 and exponent.shape in [(), (1,)]:
            # scalar exponent → single gradient value (represented as shape (1,))
            exponent_grad_val = (local_all * grad_output_b).sum()
            exponent_grad = np.array([exponent_grad_val], dtype=grad_output.dtype)
        else:
            exponent_grad = local_all * grad_output_b

        return base_grad, exponent_grad


class Dot(Function):
    @staticmethod
    def forward(ctx: "Dot", x: Tensor, y: ValueLike) -> np.ndarray:
        ctx.save_for_backward(x, y)
        x, y = Function._unwrap_args(x, y)
        return np.matmul(x, y)

    @staticmethod
    def backward(ctx: "Dot", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.saved_tensors
        _assert_allowed_backward_call(x, y)
        x, y = Function._unwrap_args(x, y)

        # Handle different shapes for broadcasting
        if x.ndim == 1 and y.ndim == 1:
            # Vector-vector dot product
            grad_x = grad_output * y
            grad_y = grad_output * x
        elif x.ndim == 1:
            # Vector-matrix multiplication
            grad_x = np.matmul(grad_output, y.T)
            grad_y = np.outer(x, grad_output)
        elif y.ndim == 1:
            # Matrix-vector multiplication
            grad_x = np.outer(grad_output, y)
            grad_y = np.matmul(x.T, grad_output)
        else:
            # Matrix-matrix multiplication
            grad_x = np.matmul(grad_output, y.T)
            grad_y = np.matmul(x.T, grad_output)

        return grad_x, grad_y


class CrossEntropyLoss(Function):
    @staticmethod
    def forward(
        ctx: "CrossEntropyLoss", logits: Tensor, targets: ValueLike
    ) -> np.ndarray:
        # we fuse the softmax and the cross entropy loss intro one op to allow for more efficient backprob
        probs = logits.softmax(-1)
        probs.item = probs.item + 1e-8
        ctx.save_for_backward(probs, targets)
        probs, targets = Function._unwrap_args(probs, targets)

        log_probs = np.log(probs)
        loss = -(targets * log_probs).sum(axis=-1)
        return loss.mean()

    @staticmethod
    def backward(ctx: "CrossEntropyLoss", grad_output: np.ndarray) -> np.ndarray:
        probs, targets = ctx.saved_tensors
        _assert_allowed_backward_call(probs, targets)
        probs, targets = Function._unwrap_args(probs, targets)
        grad_logits = (probs - targets) * grad_output

        return grad_logits

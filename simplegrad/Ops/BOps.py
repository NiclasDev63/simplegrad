from typing import Tuple
import numpy as np
from .Function import Function
from .UOps import Softmax


class Add(Function):
    @staticmethod
    def forward(ctx: "Add", x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx: "Add", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.saved_tensors
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: "Sub", x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x - y

    @staticmethod
    def backward(ctx: "Sub", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: "Mul", x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx: "Mul", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_in, y_in = ctx.saved_tensors
        x_grad = grad_output * y_in
        y_grad = grad_output * x_in
        return x_grad, y_grad


class Div(Function):
    @staticmethod
    def forward(ctx: "Div", x: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = y + 1e-8
        ctx.save_for_backward(x, y)
        return x / y

    @staticmethod
    def backward(ctx: "Div", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_in, y_in = ctx.saved_tensors

        x_grad = (1 / y_in) * grad_output
        y_grad = -x_in * (1 / (y_in**2)) * grad_output

        return x_grad, y_grad


class Pow(Function):
    @staticmethod
    def forward(ctx: "Pow", base: np.ndarray, exponent: np.ndarray) -> np.ndarray:
        assert (
            len(exponent.shape) == 1 and exponent.shape[0] == 1
        ), f"Expected exponent to be of shape (1,), but got: {exponent.shape}"
        ctx.save_for_backward(base, exponent)
        return np.pow(base, exponent)

    @staticmethod
    def backward(ctx: "Pow", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        base, exponent = ctx.saved_tensors

        base_grad = exponent * np.pow(base, exponent - 1) * grad_output

        # Compute exponent gradient only where base > 0, otherwise set to 0
        exponent_grad = np.zeros_like(grad_output)
        positive_base_mask = base > 0
        exponent_grad[positive_base_mask] = (
            np.pow(
                base[positive_base_mask], exponent[0]
            )  # Use exponent[0] since it's a scalar
            * np.log(base[positive_base_mask])
            * grad_output[positive_base_mask]
        )

        return base_grad, exponent_grad


class Dot(Function):
    @staticmethod
    def forward(ctx: "Dot", x: np.ndarray, y: np.ndarray) -> np.ndarray:
        ctx.save_for_backward(x, y)
        return np.matmul(x, y)

    @staticmethod
    def backward(ctx: "Dot", grad_output: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = ctx.saved_tensors

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
        ctx: "CrossEntropyLoss", logits: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        # we fuse the softmax and the cross entropy loss intro one op to allow for more efficient backprob
        probs = Softmax._forward(logits, axis=-1)
        probs = probs + 1e-8
        ctx.save_for_backward(probs, targets)
        log_probs = np.log(probs)
        loss = -(targets * log_probs).sum(axis=-1)
        return loss.mean()

    @staticmethod
    def backward(ctx: "CrossEntropyLoss", grad_output: np.ndarray) -> np.ndarray:
        probs, targets = ctx.saved_tensors

        grad_logits = (probs - targets) * grad_output

        return grad_logits

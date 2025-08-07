import numpy as np
from typing import Sequence, Union, Optional, Tuple
from simplegrad.Ops.Function import Function
from utils import unbroadcast, handle_axis


class Tensor:
    def __init__(
        self,
        item: Union["Tensor", np.ndarray, Sequence[Union[int, float]], int, float],
        requires_grad: bool = True,
    ) -> None:
        self.item: np.ndarray = item
        self.requires_grad = requires_grad

        # unwrap tensors
        if isinstance(self.item, Tensor):
            self.item = self.item.item
        if not isinstance(self.item, np.ndarray):
            self.item = np.array(self.item)

        # make sure there are no scalar values inside item to prevent
        # edge case handling
        if len(self.item.shape) == 0:
            self.item = np.array([self.item])

        self.item = self.item.astype(np.float32)

        self.grad: Optional[np.ndarray] = None
        self._ctx: Optional["Function"] = None

    def zero_grad(self):
        self.grad = None
        # also clear context to clean up the graph (and avoid maximum recursion depth exceeded error)
        self._ctx = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.item.shape

    def backward(self, implicit_fill: bool = True) -> None:
        if self._ctx is None or not self.requires_grad:
            return

        if self.grad is None and implicit_fill:
            self.grad = np.ones_like(self.item)

        assert (
            self.grad is not None
        ), "gradients are not available. make sure you set implicit_fill=True to implicitly create them"

        grads = self._ctx.backward(self._ctx, self.grad)

        if len(self._ctx.parents) == 1:
            grads = [grads]
        for p, g in zip(self._ctx.parents, grads):
            if p.shape != g.shape:
                g = unbroadcast(g, p.shape)
            if p.requires_grad == False:
                continue
            p.grad = g if p.grad is None else p.grad + g
            p.backward(implicit_fill=False)

    def add(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __add__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.add(other)

    def sub(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __sub__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.sub(other)

    def mul(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __mul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.mul(other)

    def div(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __truediv__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.div(other)

    def pow(self, exponent: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __pow__(self, exponent: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.pow(exponent)

    def sum(self, axis: Optional[Union[int, Tuple[int, int]]] = None) -> "Tensor":
        return self._sum(axis=axis)

    def reshape(self, shape: Tuple[int, int]) -> "Tensor":
        return self._reshape(shape=shape)

    def mean(self, axis: Optional[Union[int, Tuple[int, int]]] = None):
        if axis is None:
            n = self.item.size
        else:
            axis = handle_axis(axis=axis, ndim=self.item.ndim)
            n = np.prod([self.item.shape[a] for a in axis])
        return self.sum(axis=axis) / n

    def relu(self) -> "Tensor": ...

    def sigmoid(self) -> "Tensor": ...

    def exp(self) -> "Tensor": ...

    def log(self) -> "Tensor": ...

    def softmax(self, axis: int = -1) -> "Tensor":
        return self._softmax(axis=axis)

    def cross_entropy_loss(
        self,
        targets: Union["Tensor", np.ndarray, Sequence[Union[int, float]], int, float],
    ):
        targets = Tensor(targets)

        probs = self.softmax()

        safe_probs = probs + 1e-8
        log_probs = safe_probs.log()
        loss = targets.dot(log_probs)
        loss = -loss
        return probs, loss

    def neg(self) -> "Tensor": ...
    def __neg__(self):
        return self.neg()

    def _get_grad_fn_repr(self) -> str:
        op_name = self._ctx.__class__.__name__
        return f"<{op_name}Backward>"

    def __repr__(self) -> str:
        return f"Tensor(item={self.item}{f", grad={self.grad}" if self.grad is not None else ""}{f", grad_fn={self._get_grad_fn_repr()}" if self._ctx else ""})"


# Import and register operations at module level
def _register_operations():
    """Register all operations with the Tensor class"""
    try:
        from simplegrad.Ops.registration import register
        from simplegrad.Ops import (
            Add,
            Sub,
            Mul,
            Div,
            Pow,
            Dot,
            Sum,
            Reshape,
            Relu,
            Sigmoid,
            Exp,
            Log,
            Neg,
            Softmax,
        )

        register("add", Add)
        register("sub", Sub)
        register("mul", Mul)
        register("div", Div)
        register("pow", Pow)
        register("dot", Dot)
        register("_sum", Sum)
        register("_reshape", Reshape)
        register("relu", Relu)
        register("sigmoid", Sigmoid)
        register("exp", Exp)
        register("log", Log)
        register("neg", Neg)
        register("_softmax", Softmax)
    except ImportError as e:
        print("Error while registering tensor ops: ", str(e))


_register_operations()

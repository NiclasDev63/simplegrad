import numpy as np
from typing import Sequence, Union, Optional, Tuple, Any

from simplegrad.Ops.Function import Function
from simplegrad.utils import unbroadcast, handle_axis


class Tensor:
    def __init__(
        self,
        item: Union["Tensor", np.ndarray, Sequence[Union[int, float]], int, float],
        *,
        requires_grad: bool = False,
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
        self._ctx: Optional[Function] = None

    def zero_grad(self):
        self.grad = None
        # also clear context to clean up the graph (and avoid maximum recursion depth exceeded error)
        self._ctx = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.item.shape

    def __getitem__(self, index: Any) -> "Tensor":
        return self._index(index=index)

    def __setitem__(
        self, index: Any, value: Union["Tensor", np.ndarray, int, float]
    ) -> None:
        # TODO: handle RHS (value) is tensor case
        # clone current tensor into new one
        old_self = Tensor(self.item.copy(), requires_grad=self.requires_grad)
        old_self._ctx = self._ctx

        # Build the copy-slice node on the old-self wrapper
        res = old_self._copy_slices(index=index, value=value)

        # Graft the new node onto this tensor so downstream ops see it
        self.item = res.item
        self.requires_grad = res.requires_grad
        self._ctx = res._ctx

    def backward(self, implicit_fill: bool = True) -> None:
        if not self.requires_grad:
            raise ValueError(
                ".backward() can not be called on a tensor with requires_grad=False"
            )

        if self._ctx is None:
            return

        if self.grad is None and implicit_fill:
            self.grad = np.ones_like(self.item)

        assert (
            self.grad is not None
        ), "gradients are not available. make sure you set implicit_fill=True to implicitly create them"

        grads = self._ctx.backward(self._ctx, self.grad)

        parents = self._ctx.parents
        if len(parents) == 1 and not isinstance(grads, tuple):
            grads = [grads]

        for p, g in zip(parents, grads):
            if p.shape != g.shape:
                g = unbroadcast(g, p.shape)
            p.grad = g if p.grad is None else p.grad + g
            p.backward(implicit_fill=False)

    def add(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __add__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.add(other)

    def __radd__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        # addition is commutative; reuse the same op
        return self.add(other)

    def sub(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __sub__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.sub(other)

    def __rsub__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor(other).sub(self)

    def mul(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __mul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.mul(other)

    def __rmul__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        # multiplication is commutative; reuse the same op
        return self.mul(other)

    def div(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __truediv__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.div(other)

    def __rtruediv__(self, other: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return Tensor(other).div(self)

    def pow(self, exponent: Union["Tensor", np.ndarray, int, float]) -> "Tensor": ...
    def __pow__(self, exponent: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        return self.pow(exponent)

    def __rpow__(self, base: Union["Tensor", np.ndarray, int, float]) -> "Tensor":
        base_t = base if isinstance(base, Tensor) else Tensor(base)
        return base_t.pow(self)

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

    def softmax(self, axis: int = -1, temperature: int = 1) -> "Tensor":
        return self._softmax(axis=axis, temperature=temperature)

    def cross_entropy_loss(
        self,
        targets: Union["Tensor", np.ndarray, Sequence[Union[int, float]], int, float],
    ) -> "Tensor": ...

    def neg(self) -> "Tensor": ...
    def __neg__(self):
        return self.neg()

    def _get_grad_fn_repr(self) -> str:
        op_name = self._ctx.__class__.__name__
        return f"<{op_name}Backward>"

    def __repr__(self) -> str:
        return f"Tensor(item={self.item}, requires_grad={self.requires_grad}{f", grad={self.grad}" if self.grad is not None else ""}{f", grad_fn={self._get_grad_fn_repr()}" if self._ctx else ""})"


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
            CrossEntropyLoss,
            Sum,
            Reshape,
            Relu,
            Sigmoid,
            Exp,
            Log,
            Neg,
            Softmax,
            Index,
            CopySlices,
        )

        register("add", Add)
        register("sub", Sub)
        register("mul", Mul)
        register("div", Div)
        register("pow", Pow)
        register("dot", Dot)
        register("cross_entropy_loss", CrossEntropyLoss)
        register("_sum", Sum)
        register("_reshape", Reshape)
        register("relu", Relu)
        register("sigmoid", Sigmoid)
        register("exp", Exp)
        register("log", Log)
        register("neg", Neg)
        register("_softmax", Softmax)
        register("_index", Index)
        register("_copy_slices", CopySlices)
    except ImportError as e:
        print("Error while registering tensor ops: ", str(e))


_register_operations()

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

        # used to check whether tensor was modified by an inplace operation
        # only necessary for gradient calculation
        self._version = 0

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
        # Clone current tensor into a wrapper representing the pre-write value
        old_self = Tensor(self.item.copy(), requires_grad=self.requires_grad)
        old_self._ctx = self._ctx
        old_self._version = self._version

        rhs: Union["Tensor", np.ndarray, int, float] = value

        if isinstance(rhs, Tensor) and self._tensor_depends_on(rhs, self):
            rhs = self._clone_graph_with_replacement(rhs, {id(self): old_self})

        # Build the copy-slice node on the old-self wrapper
        res = old_self._copy_slices(index=index, value=rhs)

        # Graft the new node onto this tensor so downstream ops see it
        self.item = res.item
        self.requires_grad = res.requires_grad
        self._ctx = res._ctx
        self._version = old_self._version

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if not self.requires_grad:
            return

        # Accumulate this call's gradient
        if grad is None:
            if self.grad is None:
                self.grad = np.ones_like(self.item)
            local_grad = self.grad
        else:
            self.grad = grad if self.grad is None else self.grad + grad
            local_grad = grad

        # Leaf tensors have no context
        if self._ctx is None:
            return

        # Compute gradients for parents and recurse
        grads = self._ctx.backward(self._ctx, local_grad)
        parents = self._ctx.parents

        if len(parents) == 1 and not isinstance(grads, tuple):
            grads = [grads]

        for parent, parent_grad in zip(parents, grads):
            if parent.shape != parent_grad.shape:
                parent_grad = unbroadcast(parent_grad, parent.shape)
            if parent.requires_grad:
                parent.backward(parent_grad)

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

    def _tensor_depends_on(self, source: "Tensor", target: "Tensor") -> bool:
        """Iteratively check if `source`'s graph reaches `target`."""
        if not isinstance(source, Tensor):
            return False
        if source is target:
            return True
        visited: set[int] = set()
        stack: list[Tensor] = [source]
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            ctx = node._ctx
            if ctx is None:
                continue
            for parent in ctx.parents:
                if parent is target:
                    return True
                stack.append(parent)
        return False

    def _clone_graph_with_replacement(
        self, tensor: "Tensor", replace_by_id: dict[int, "Tensor"]
    ) -> "Tensor":
        """Return a clone of `tensor`'s graph, replacing nodes by identity.

        Any node whose `id(node)` appears in `replace_by_id` is replaced by the
        provided Tensor. Other nodes are cloned
        """

        memo: dict[int, Tensor] = {}

        def clone(node: "Tensor") -> "Tensor":
            # Direct replacement
            replacement = replace_by_id.get(id(node))
            if replacement is not None:
                return replacement

            # Already cloned
            cached = memo.get(id(node))
            if cached is not None:
                return cached

            # Clone tensor payload
            node_clone = Tensor(node.item.copy(), requires_grad=node.requires_grad)
            memo[id(node)] = node_clone

            # Leaf tensors have no context
            if node._ctx is None:
                return node_clone

            # Clone parents and recreate op context
            parents_clone = tuple(clone(p) for p in node._ctx.parents)
            ctx_cls = node._ctx.__class__
            new_ctx = ctx_cls(
                *parents_clone, in_place=getattr(node._ctx, "in_place", False)
            )
            new_ctx.saved_tensors = list(node._ctx.saved_tensors)
            node_clone._ctx = new_ctx
            return node_clone

        return clone(tensor)


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

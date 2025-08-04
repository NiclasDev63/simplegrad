import numpy as np
from typing import Sequence, TypeAlias, Union
from numpy.typing import DTypeLike, NDArray
from enum import Enum

Number: TypeAlias = Union[int, float, bool]


class OP(Enum):
    ADD = "add"
    SUB = "subtract"
    DIV = "divide"
    MUL = "multiply"


class UOP(Enum):
    POW = "power"


class Tensor:
    def __init__(
        self,
        value: Sequence[Number],
        dtype: DTypeLike = np.float32,
        _children: Sequence["Tensor"] = (),
        _op: Union[OP, UOP, None] = None,
    ):
        self.dtype = dtype
        self._item = np.array(value).astype(self.dtype)
        self.grad = 0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    @property
    def item(self) -> NDArray:
        return self._item

    def to(self, dtype: DTypeLike):
        self._item = self.item.astype(dtype)
        self.dtype = dtype

    def _check_and_convert(self, tensor: "Tensor" | Sequence[Number]) -> "Tensor":
        if not isinstance(tensor, type(self)):
            tensor = Tensor(tensor, self.dtype)
        return tensor

    def __add__(self, other: "Tensor" | Sequence[Number]):
        other_tensor = self._check_and_convert(other)
        out = Tensor(
            self.item + other_tensor.item,
            dtype=self.dtype,
            _children=[self, other],
            _op=OP.ADD,
        )

        def _backward():
            self.grad += out.grad
            other_tensor.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: "Tensor" | Sequence[Number]):
        return self + other

    def __sub__(self, other: "Tensor" | Sequence[Number]):
        other_tensor = self._check_and_convert(other)
        out = Tensor(
            self.item - other_tensor.item,
            dtype=self.dtype,
            _children=[self, other],
            _op=OP.SUB,
        )

        def _backward():
            self.grad -= out.grad
            other_tensor.grad -= out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Tensor" | Sequence[Number]):
        other_tensor = self._check_and_convert(other)
        out = Tensor(
            self.item * other_tensor.item,
            dtype=self.dtype,
            _children=[self, other],
            _op=OP.MUL,
        )

        def _backward():
            self.grad += other_tensor.item * out.grad
            other_tensor.grad += self.item * out.grad

        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only support int and float powers"
        out = Tensor(
            np.pow(self.item, other), dtype=self.dtype, _children=[self], _op=UOP.POW
        )

        def __backward():
            self.grad += (other * np.pow(self.item, other - 1)) * out.grad

        out._backward = __backward

        return out

    def backward(self):
        nodes: list["Tensor"] = []
        visited = set()

        def build_graph(v: "Tensor"):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_graph(child)
                nodes.append(v)

        build_graph(self)

        self.grad = 1
        for v in reversed(nodes):
            v._backward()

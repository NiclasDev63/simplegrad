from functools import partialmethod
import numpy as np


class Tensor:
    def __init__(self, item):
        self.item = item
        if not isinstance(self.item, np.ndarray):
            self.item = np.array(item)

        self.grad = None
        self._ctx = None

    def shape(self):
        return self.item.shape

    def backward(self, implicit_fill=True):
        if self._ctx is None:
            return

        if self.grad is None and implicit_fill:
            shape = self.item.shape
            if len(shape) != 1 or shape[0] > 1:
                raise TypeError(
                    "grad can be implicitly created only for scalar outputs"
                )
            self.grad = 1

        assert self.grad is not None, (
            "gradients are not available. make sure you set implicit_fill=True to implicitly create them"
        )

        grads = self._ctx.backward(self._ctx, self.grad)
        print("GRADS: ", grads)
        if len(self._ctx.parents) == 1:
            grads = [grads]
        for p, g in zip(self._ctx.parents, grads):
            # if g.shape != p.shape:
            #     raise ValueError(
            #         f"Grad shape must match tensor shape in {self._get_grad_fn_repr()}, {g.shape} != {p.shape}"
            #     )
            p.grad = g
            p.backward(implicit_fill=False)

    def __add__(self, other: np.ndarray):
        return self.add(other)

    def __sub__(self, other: np.ndarray):
        return self.sub(other)

    def __mul__(self, other: np.ndarray):
        return self.mul(other)

    def _get_grad_fn_repr(self):
        op_name = self._ctx.__class__.__name__
        return f"<{op_name}Backward>"

    def __repr__(self):
        return f"Tensor(item={self.item}, grad={self.grad}, grad_fn={self._get_grad_fn_repr()})"


class Function:
    def __init__(self, *tensors):
        self.parents = tensors
        self.saved_tensors = []

    def save_for_backward(self, *x):
        self.saved_tensors.extend(x)

    def apply(self: Tensor, operation, *args):
        ctx = operation(self, *args)
        res = Tensor(operation.forward(ctx, self.item, *[t.item for t in args]))
        res._ctx = ctx
        return res


def register(name, fxn):
    return setattr(Tensor, name, partialmethod(fxn.apply, fxn))


class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x + y

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output


register("add", Add)


class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x - y

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output, -grad_output


register("sub", Sub)


class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y

    @staticmethod
    def backward(ctx, grad_output):
        x_in, y_in = ctx.saved_tensors
        x_grad = grad_output * y_in
        y_grad = grad_output * x_in
        return x_grad, y_grad


register("mul", Mul)

a = Tensor([5])
b = Tensor([10])
c =  Tensor([2])
res = a * b * c
res.backward()
print("A grad: ", a.grad)
print("B grad: ", b.grad)
print("RES: ", res.item)

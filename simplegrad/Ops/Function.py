from typing import Union, List, Any, Tuple, TYPE_CHECKING
import numpy as np

ValueLike = Union[np.ndarray, float, int]


if TYPE_CHECKING:
    from simplegrad.Tensor.Tensor import Tensor


class Function:
    """Base class for all operations available on the Tensor object"""

    def __init__(self, *tensors: "Tensor", in_place: bool = False) -> None:
        self.parents: Tuple["Tensor", ...] = tensors
        self.saved_tensors: List[Any] = []
        self.in_place = in_place

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    @staticmethod
    def _unwrap_args(*data: Union["Tensor", np.ndarray, int, float]) -> list["Tensor"]:
        from simplegrad import Tensor

        return [d.item if isinstance(d, Tensor) else d for d in data]

    @staticmethod
    def _unwrap_kwargs(
        **named_data: Union["Tensor", np.ndarray, int, float]
    ) -> dict[str, "Tensor"]:
        from simplegrad import Tensor

        return {
            k: t.item if isinstance(t, Tensor) else t for k, t in named_data.items()
        }

    @staticmethod
    def _compute_requires_grad(curr: "Tensor", parents: list["Tensor"]) -> bool:
        """
        Internal function to compute whether the resulting Tensor requires gradients.
        True if at least one Tensor involved in this operation has requires_grad=True
        """
        if curr.requires_grad:
            return True
        return any(item.requires_grad for item in parents)

    @staticmethod
    def _get_parents(
        fn_args: list["Tensor"], fn_kwargs: dict[str, "Tensor"]
    ) -> list["Tensor"]:
        from simplegrad import Tensor

        parents = []
        for arg in fn_args:
            if isinstance(arg, Tensor) and arg.requires_grad:
                parents.append(arg)

        for kwarg in fn_kwargs.values():
            if isinstance(kwarg, Tensor) and kwarg.requires_grad:
                parents.append(kwarg)
        return parents

    def apply(
        self: "Tensor",
        op_fn: "Function",
        *fn_args: Union["Tensor", np.ndarray, int, float],
        **fn_kwargs: Union["Tensor", np.ndarray, int, float]
    ) -> "Tensor":
        from simplegrad import Tensor

        # only add parents which requires gradients to the graph to minimize memory footprint
        parents = op_fn._get_parents(fn_args=fn_args, fn_kwargs=fn_kwargs)

        is_in_place = bool(getattr(op_fn, "_in_place", False))
        ctx = op_fn(self, *parents, in_place=is_in_place)

        requires_grad = op_fn._compute_requires_grad(curr=self, parents=parents)

        # important that we unwrap the args and kwargs AFTER registering them as parents
        # and checking if the resulting tensor should have requires_grad=True
        fn_args = op_fn._unwrap_args(*fn_args)
        fn_kwargs = op_fn._unwrap_kwargs(**fn_kwargs)
        res = Tensor(
            item=op_fn.forward(ctx, self.item, *fn_args, **fn_kwargs),
            requires_grad=requires_grad,
        )
        res._ctx = ctx
        return res

    @staticmethod
    def forward(
        ctx: "Function", self: "Tensor", *args: Any, **kwargs: Any
    ) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def backward(
        ctx: "Function", grad_output: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError()

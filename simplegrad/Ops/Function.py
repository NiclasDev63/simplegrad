from threading import stack_size
from typing import Union, List, Any, Tuple, TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from simplegrad import Tensor


ValueLike = Union["Tensor", np.ndarray, float, int]


class Function:
    """Base class for all operations available on the Tensor object"""

    def __init__(self, *tensors: "Tensor", in_place: bool = False) -> None:
        self.parents: Tuple["Tensor", ...] = tensors
        self.saved_tensors: List[Any] = []
        self.in_place = in_place

    def save_for_backward(self, *x: Any) -> None:
        from simplegrad import Tensor

        # Save tensors and their version counters to detect in-place changes
        for obj in x:
            if isinstance(obj, Tensor):
                self.saved_tensors.append((obj._version, obj))
            else:
                self.saved_tensors.append(obj)

    @staticmethod
    def is_pair(x: object) -> bool:
        from simplegrad import Tensor

        return (
            isinstance(x, tuple)
            and len(x) == 2
            and isinstance(x[0], int)
            and isinstance(x[1], Tensor)
        )

    @staticmethod
    def _unwrap_args(*data: Union["Tensor", np.ndarray, int, float]) -> list["Tensor"]:
        from simplegrad import Tensor

        result = []
        for d in data:
            # Only treat (version, Tensor) as a saved-tensor pair; leave normal tuples (e.g. indices) intact
            if Function.is_pair(d):
                result.append(d[1].item)
            elif isinstance(d, Tensor):
                result.append(d.item)
            else:
                result.append(d)
        return result[0] if len(result) == 1 else result

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
        # important to update the current tensor version AFTER the new op

        res = Tensor(
            item=op_fn.forward(ctx, self, *fn_args, **fn_kwargs),
            requires_grad=requires_grad,
        )

        # Bump version AFTER forward for in-place ops so RHS saved tensors see pre-bump version
        if is_in_place:
            self._version = self._version + 1
        res._ctx = ctx
        res._version = self._version
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

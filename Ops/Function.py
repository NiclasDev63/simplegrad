from typing import Union, List, Any, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from Tensor.Tensor import Tensor


class Function:
    """Base class for all operations available on the Tensor object"""

    def __init__(self, *tensors: "Tensor") -> None:
        self.parents: Tuple["Tensor", ...] = tensors
        self.saved_tensors: List[Any] = []

    def save_for_backward(self, *x: Any) -> None:
        self.saved_tensors.extend(x)

    @staticmethod
    def _args_to_tensor(
        *data: Union["Tensor", np.ndarray, int, float]
    ) -> list["Tensor"]:
        from Tensor.Tensor import Tensor

        return [d if isinstance(d, Tensor) else Tensor(d) for d in data]

    @staticmethod
    def _kwargs_to_tensor(
        **named_data: Union["Tensor", np.ndarray, int, float]
    ) -> dict[str, "Tensor"]:
        from Tensor.Tensor import Tensor

        return {
            k: t if isinstance(t, Tensor) else Tensor(t) for k, t in named_data.items()
        }

    def apply(
        self: "Tensor",
        op_fn: "Function",
        *fn_args: Union["Tensor", np.ndarray, int, float],
        **fn_kwargs: Union["Tensor", np.ndarray, int, float]
    ) -> "Tensor":
        from Tensor.Tensor import Tensor

        fn_args = op_fn._args_to_tensor(*fn_args)
        fn_kwargs = op_fn._kwargs_to_tensor(**fn_kwargs)
        ctx = op_fn(self, *fn_args)
        res = Tensor(
            op_fn.forward(
                ctx,
                self.item,
                *[t.item for t in fn_args],
                **{k: t.item for k, t in fn_kwargs.items()}
            ),
        )
        res._ctx = ctx
        return res

    @staticmethod
    def forward(ctx: "Function", *inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def backward(
        ctx: "Function", grad_output: np.ndarray
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError()

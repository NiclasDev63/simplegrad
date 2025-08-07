from functools import partialmethod
from .Function import Function


def register(name: str, op_fn: Function) -> None:
    """Utility function to register a new OP to the Tensor object"""
    # make sure we pass the current OP function as first argument using partialmethod
    # to avoid having to do this when calling the Tensor function
    from simplegrad.Tensor.Tensor import Tensor
    return setattr(Tensor, name, partialmethod(op_fn.apply, op_fn)) 
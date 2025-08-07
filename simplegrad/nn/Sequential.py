from typing import Any, Callable

from simplegrad import Tensor
from .Module import Module


class Sequential(Module):
    """Wrapper class to stack multiple modules and functions together in a sequential order"""

    def __init__(self, modules: list[Module | Callable[Any, Tensor]]):
        super().__init__()
        self.modules = modules

    def forward(self, input):
        for module in self.modules:
            input = module(input)
        return input

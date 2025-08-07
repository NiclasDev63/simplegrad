from abc import ABC, abstractmethod
from typing import Iterator, Dict

from .Parameter import Parameter
from simplegrad import Tensor


class Module(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Tensor]:
        for param_dict in self.named_parameters():
            param = list(param_dict.values())[0]
            yield param

    def named_parameters(self) -> Iterator[Dict[str, Tensor]]:
        for attr_name in dir(self):
            attr = getattr(self, attr_name)

            if isinstance(attr, Parameter):
                yield {attr_name: attr}
            elif isinstance(attr, Module):
                yield from self._yield_nested_params(attr, f"{attr_name}.")
            elif isinstance(attr, list):
                yield from self._yield_list_params(attr, attr_name)

    def _yield_nested_params(
        self, module: "Module", prefix: str
    ) -> Iterator[Dict[str, Tensor]]:
        """Helper method to yield parameters from nested modules."""
        for param_dict in module.named_parameters():
            nested_name = list(param_dict.keys())[0]
            nested_param = param_dict[nested_name]
            yield {prefix + nested_name: nested_param}

    def _yield_list_params(
        self, attr_list: list, attr_name: str
    ) -> Iterator[Dict[str, Tensor]]:
        """Helper method to yield parameters from modules stored in a list."""
        modules = [item for item in attr_list if isinstance(item, Module)]
        for i, module in enumerate(modules):
            module_prefix = f"{attr_name}.{i}."
            yield from self._yield_nested_params(module, module_prefix)

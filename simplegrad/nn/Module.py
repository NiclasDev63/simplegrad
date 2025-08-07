from abc import ABC, abstractmethod

from .Parameter import Parameter
from simplegrad.Tensor import Tensor


class Module(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def parameters(self) -> list[Tensor]:
        params = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                params.append(attr)
            elif isinstance(attr, Module):
                params.extend(attr.parameters())

        return params

    def named_parameters(self) -> dict[Tensor]:
        params = {}
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, Parameter):
                params[attr_name] = attr
            elif isinstance(attr, Module):
                named_params = attr.named_parameters()
                prefix = f"{attr_name}."
                for name, param in named_params.items():
                    params[prefix + name] = param
        return params

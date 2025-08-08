from abc import ABC, abstractmethod
from typing import Iterator, Dict

import os
import pickle
import warnings

import numpy as np

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

    def state_dict(self) -> Dict[str, np.ndarray]:
        """Return a mapping from parameter names to raw numpy arrays."""
        state: Dict[str, np.ndarray] = {}
        for param_dict in self.named_parameters():
            name, param = next(iter(param_dict.items()))
            # store raw data only to avoid serializing autograd graphs
            state[name] = param.item.copy()
        return state

    def save_state_dict(self, path: str = "checkpoints/model.pickle"):
        """Save the current state dict to a file using pickle."""

        if os.path.exists(path):
            raise ValueError("Can not save model state, path already exists")

        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(self.state_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state_dict(self, path: str, strict: bool = True):
        """Load a state dict from a pickle file and assign to parameters.

        - strict=True will raise if shapes mismatch or keys are missing/unexpected.
        """
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        if not isinstance(loaded, dict):
            raise TypeError("Loaded state must be a dict[str, np.ndarray].")

        current_names = []
        for param_dict in self.named_parameters():
            name, param = next(iter(param_dict.items()))
            current_names.append(name)
            if name not in loaded:
                if strict:
                    raise KeyError(f"Missing key in loaded state: {name}")
                else:
                    continue

            value = loaded[name]
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=np.float32)

            if param.item.shape != value.shape:
                if strict:
                    raise ValueError(
                        f"Shape mismatch for '{name}': model {param.item.shape} vs loaded {value.shape}"
                    )
                else:
                    warnings.warn(
                        f"Shape mismatch for '{name}': model {param.item.shape} vs loaded {value.shape}. Skipping assignment.",
                        UserWarning,
                    )
                    continue

            param.item = value.astype(np.float32)

        if strict:
            unexpected = set(loaded.keys()) - set(current_names)
            if unexpected:
                raise KeyError(f"Unexpected keys in loaded state: {sorted(unexpected)}")

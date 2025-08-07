from abc import ABC, abstractmethod

from simplegrad.Tensor import Tensor


class Optimizer(ABC):

    def __init__(self, params: list[Tensor], lr: float):
        self.lr = lr
        self.params = self._register_params(params)

    def _register_params(self, params: list[Tensor]) -> list[Tensor]:
        grad_params = []
        for param in params:
            if param.requires_grad:
                grad_params.append(param)

        assert len(grad_params) > 0, "no parameters which requires grads where provided"
        return grad_params

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

    @abstractmethod
    def step(self):
        pass

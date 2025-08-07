from .Optimizer import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2

        self.mv1 = []
        self.mv2 = []
        self._init_moments()

        self.eps = eps

        if 0 > beta1 > 0.9999 or 0 > beta2 > 0.9999:
            raise ValueError("Betas have to be between 0 and 1 (exclusive)")

    def _init_moments(self):
        for param in self.params:
            zeros = np.zeros_like(param)
            self.mv1.append(zeros)
            self.mv2.append(zeros)

    def step(self):
        for t, param in enumerate(self.params):
            step_t = t + 1

            self.mv1[t] = self.beta1 * self.mv1[t] + (1 - self.beta1) * param.grad
            self.mv2[t] = self.beta2 * self.mv2[t] + (1 - self.beta2) * param.grad**2

            mv1_bias_correct = self.mv1[t] / (1 - self.beta1**step_t)
            mv2_bias_correct = self.mv2[t] / (1 - self.beta2**step_t)

            param.item = param.item - self.lr * (
                mv1_bias_correct / (np.sqrt(mv2_bias_correct) + self.eps)
            )

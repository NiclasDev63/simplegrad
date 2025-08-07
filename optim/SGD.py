from .Optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, params, lr=0.01):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.item = param.item - self.lr * param.grad

from .Optimizer import Optimizer


# TODO: add momentum
class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)

    def step(self):
        for param in self.params:
            param.item = param.item - self.lr * param.grad

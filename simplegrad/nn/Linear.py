from .weight_init import xavier_init
from simplegrad.Tensor import Tensor
import numpy as np
from .Module import Module
from .Parameter import Parameter


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, use_bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias

        self.weight = Parameter(Tensor(xavier_init(self.in_dim, self.out_dim)))
        
        if self.use_bias:
            self.bias = Parameter(Tensor(np.zeros(self.out_dim)))

    def forward(self, x: Tensor):
        out = x.dot(self.weight)
        if self.use_bias:
            return out + self.bias
        return out

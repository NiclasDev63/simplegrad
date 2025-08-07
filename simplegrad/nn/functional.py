from typing import Optional, Tuple, Union
from simplegrad import Tensor
import numpy as np


def relu(x: Tensor):
    return x.relu()


def softmax(x: Tensor, axis: int = -1):
    return x.softmax(axis=axis)


def sigmoid(x: Tensor):
    return x.sigmoid()


def log(x: Tensor):
    return x.log()


def exp(x: Tensor):
    return x.exp()


def sum(x: Tensor, axis: Optional[Union[int, Tuple[int, int]]] = None):
    return x.sum(axis=axis)


def dot(x: Tensor, y: Tensor):
    return x.dot(y)


def pow(base: Tensor, exponent: Union["Tensor", np.ndarray, int, float]):
    return base.pow(exponent=exponent)

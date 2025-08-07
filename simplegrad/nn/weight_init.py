import numpy as np


def xavier_init(in_dim, out_dim):
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return np.random.normal(0, std, (in_dim, out_dim)).astype(np.float32)

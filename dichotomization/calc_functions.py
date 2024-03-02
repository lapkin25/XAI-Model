import math
import numpy as np


def stable_sigmoid(z):
    if z >= 0:
        return 1 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (np.exp(z) + 1)


def inv_sigmoid(p):
    return math.log(p / (1 - p))

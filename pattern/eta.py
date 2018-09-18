import numpy as np

from pattern.constant import W1


def eta(x, xi):
    return W1 * (1 - np.cos(xi * x * np.pi))


def deta(x, xi):
    return W1 * np.sin(xi * x * np.pi) * xi * np.pi

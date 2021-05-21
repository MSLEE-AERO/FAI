import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z):
    if z > 0:
        return z
    else:
        return 0.01 * z


def grad_sigmoid(z):
    return z * (1 - z)


def grad_softmax(z):
    m = len(z)
    grad = np.zeros(shape=[m, m])
    for i in range(m):
        sigma_i = softmax(z)[i]
        for j in range(m):
            sigma_j = softmax(z)[j]
            grad[i, j] = sigma_i * (kronecker_delta(i, j) - sigma_j)
    return grad


def grad_relu(z):
    if z > 0:
        return 1
    else:
        return 0


def grad_leaky_relu(z):
    if z > 0:
        return 1
    else:
        return 0.01


def kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0

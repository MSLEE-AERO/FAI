import numpy as np


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def softmax(z):
    z = z - np.max(z, keepdims=True, axis=1)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z):
    return np.maximum(0.01*z, z)


def grad_sigmoid(z):
    return z * (1 - z)


def grad_softmax(z):
    n = len(z)
    grad = np.zeros(shape=(n, n))
    for i in range(n):
        sigma_i = softmax(z)[i]
        for j in range(n):
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


def activation_function(z, name):
    if name == 'sigmoid':
        return sigmoid(z)
    elif name == 'softmax':
        return softmax(z)
    elif name == 'relu':
        return relu(z)
    elif name == 'leaky_relu':
        return leaky_relu(z)
    else:
        print('invalid activation function name')
        quit()
        return 404


def grad_actiavtion_function(z, name):
    if name == 'sigmoid':
        return grad_sigmoid(z)
    elif name == 'softmax':
        return grad_softmax(z)
    elif name == 'relu':
        return grad_relu(z)
    elif name == 'leaky_relu':
        return grad_leaky_relu(z)
    else:
        print('invalid activation function name')
        quit()
        return 404
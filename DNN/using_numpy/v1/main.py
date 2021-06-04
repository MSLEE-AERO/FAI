import numpy as np

import header as h
import matplotlib.pyplot as plt

num_of_data_train = 1000
num_of_data_test = int(num_of_data_train * 0.2)
#
num_of_features = 2
#
x_train = np.random.randn(num_of_data_train, num_of_features)
x_test = np.random.randn(num_of_data_test, num_of_features)
y_train = np.zeros(shape=(num_of_data_train, 1))
y_test = np.zeros(shape=(num_of_data_test, 1))
#
for m in range(num_of_data_train):
    if m > 0.5 * num_of_data_train:
        y_train[m] = 1

np.random.shuffle(x_train)
np.random.shuffle(y_train)
#
for m in range(num_of_data_test):
    if m > 0.5 * num_of_data_test:
        y_test[m] = 1
#
np.random.shuffle(x_test)
np.random.shuffle(y_test)
#
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train[:, 0])
plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test[:, 0])
plt.show()
#
layers = [num_of_features, 20, 16, 1]
num_of_layers = len(layers)
#
alpha = 0.001
lamb = 0.001
#
params = {}
for i in range(1, num_of_layers):
    params["w" + str(i)] = np.random.rand(layers[i - 1], layers[i]) * 0.001
    params["b" + str(i)] = np.zeros((1, layers[i]))


def grad_loss(a, y, m, loss_function):
    if loss_function == 'cross_entropy':
        return - (y / a) / m
    elif loss_function == 'MSE':
        return (a - y) / m


def calc_loss(y, a, m, loss_function):
    if loss_function == "cross_entropy":
        return np.sum(-(y * np.log(a))) / m
    elif loss_function == "MSE":
        return np.sum(np.square(y - a)) / (2 * m)


def sigmoid(z):
    return 1. / (1. + np.exp(-z))


def softmax(z):
    z = z - np.max(z, keepdims=True, axis=1)
    return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def leaky_relu(z):
    return np.maximum(0.01 * z, z)


def grad_sigmoid(dLda, z):
    sigma = sigmoid(z)
    return dLda * sigma * (1 - sigma)


def grad_softmax(dLda, z):
    a = softmax(z)

    dz = np.zeros(dLda.shape)
    (m, dim) = dLda.shape
    for k in range(m):
        middle = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    middle[i, j] = a[k, i] * (1 - a[k, i])
                else:
                    middle[i, j] = -(a[k, i] * a[k, j])
        dz[k, :] = np.matmul(dLda[k, :], middle)
    return dz


def grad_relu(dLda, z):
    dz = np.ones(z.shape)
    dz[z > 0] = 0.01
    return dLda * dz


def grad_leaky_relu(dLda, z):
    dz = np.ones(z.shape)
    dz[z < 0] = 0.01
    return dLda * dz


def kronecker_delta(i, j):
    if i == j:
        return 1.0
    else:
        return 0.0


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


def grad_activation_function(dLda, z, name):
    if name == 'sigmoid':
        return grad_sigmoid(dLda, z)
    elif name == 'softmax':
        return grad_softmax(dLda, z)
    elif name == 'relu':
        return grad_relu(dLda, z)
    elif name == 'leaky_relu':
        return grad_leaky_relu(dLda, z)
    else:
        print('invalid activation function name')
        quit()
        return 404


h.grads = {}
h.a = {}
h.z = {}
if __name__ == "__main__":
    epochs = 100
    num_of_iterations = 1000
    activation_func = 'sigmoid'
    loss_func = 'cross_entropy'
    total_loss = []
    for i in range(epochs):
        for j in range(num_of_iterations):
            x = x_train
            for ilayer in range(1, num_of_layers):
                w = params["w" + str(ilayer)]
                b = params["b" + str(ilayer)]
                h.z[ilayer] = np.matmul(x, w) + b
                if ilayer == num_of_layers - 1:
                    h.a[ilayer] = activation_function(h.z[ilayer], 'sigmoid')
                else:
                    h.a[ilayer] = activation_function(h.z[ilayer], activation_func)
                x = h.a[ilayer]

            for ilayer in range(num_of_layers - 1, 0, -1):
                if ilayer == num_of_layers - 1:
                    dLda = grad_loss(h.a[ilayer], y_train, num_of_data_train, loss_func)

                    dLdz = grad_activation_function(dLda, h.z[ilayer], 'softmax')

                    dzdw = h.a[ilayer - 1]
                    h.grads["dw" + str(ilayer)] = np.matmul(dzdw.transpose(), dLdz)
                    h.grads["db" + str(ilayer)] = np.mean(dLdz, axis=0, keepdims=True)
                else:
                    dzda = params["w" + str(ilayer + 1)]
                    dLda = np.matmul(dLdz, dzda.transpose())
                    dLdz = grad_activation_function(dLda,h.z[ilayer], activation_func)
                    if ilayer == 1:
                        dzdw = x_train
                    else:
                        dzdw = h.a[ilayer - 1]
                    h.grads["dw" + str(ilayer)] = np.matmul(dzdw.transpose(), dLdz)
                    h.grads["db" + str(ilayer)] = np.mean(dLdz, axis=0, keepdims=True)
            for ilayer in range(1, num_of_layers):
                params["w" + str(ilayer)] = params["w" + str(ilayer)] - alpha * h.grads["dw" + str(ilayer)]
                params["b" + str(ilayer)] = params["b" + str(ilayer)] - alpha * h.grads["db" + str(ilayer)]

            if j % 100 == 0:
                loss = calc_loss(y_train, h.a[num_of_layers - 1], num_of_data_train, loss_func)
                total_loss.append(loss)
                print(f"epoch: {i}, iteration: {j} loss: {loss}")

    plt.figure()
    plt.plot(total_loss)
    plt.xlabel("num of iterations")
    plt.ylabel("loss")
    plt.title("DNN")
    plt.show()

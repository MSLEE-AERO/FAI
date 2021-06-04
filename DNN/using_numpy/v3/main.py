import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def init_params(dims_of_layers):
    num_of_layers = len(dims_of_layers)
    params = {}
    for i in range(1, num_of_layers):
        params["w" + str(i)] = np.random.randn(dims_of_layers[i - 1], dims_of_layers[i]) * 0.001
        params["b" + str(i)] = np.zeros((1, dims_of_layers[i]))
    return params


def linear(a, w, b):
    linear_cache = [a, w, b]
    z = np.matmul(a, w) + b
    return z, linear_cache


def linear_grad(dz, linear_cache):
    [a, w, b] = linear_cache
    grads = {"dw": np.matmul(a.T, dz),
             "db": np.mean(dz, axis=0, keepdims=True),
             "da": np.matmul(dz, w.T)}
    return grads


def sigmoid(z):
    activation_cache = [z]
    a = 1 / (1 + np.exp(-z))
    return a, activation_cache


def sigmoid_grad(da, activation_cache):
    [z] = activation_cache
    a = 1 / (1 + np.exp(-z))
    return da * a * (1 - a)


def relu(z):
    activation_cache = [z]
    a = np.maximum(0, z)
    return a, activation_cache


def relu_gradient(da, activation_cache):
    [z] = activation_cache
    dz = np.ones(z.shape)
    dz[z < 0] = 0
    return da * dz


def leaky_relu(z):
    activation_cache = [z]
    a = np.maximum(0.01 * z, z)
    return a, activation_cache


def leaky_relu_gradient(da, activation_cache):
    [z] = activation_cache
    dz = np.ones(da.shape)
    dz[z < 0] = 0.01
    return da * dz


def softmax(z):
    activation_cache = [z]
    z = z - np.max(z, axis=1, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    return a, activation_cache


def softmax_gradient(dloss, activation_cache):
    [z] = activation_cache
    z = z - np.max(z, axis=1, keepdims=True)
    a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    dz = np.zeros(dloss.shape)
    (m, dim) = dloss.shape
    for k in range(m):
        middle_matrix = np.zeros((dim, dim))
        for i in range(dim):
            for j in range(dim):
                if i == j:
                    middle_matrix[i, j] = a[k, i] * (1 - a[k, i])
                else:
                    middle_matrix[i, j] = -(a[k, i] * a[k, j])
        dz[k, :] = np.matmul(dloss[k, :], middle_matrix)
    return dz


def single_forward(a, w, b, activation):
    z, linear_cache = linear(a, w, b)
    if activation == "relu":
        a, activation_cache = relu(z)
    elif activation == "leaky_relu":
        a, activation_cache = leaky_relu(z)
    elif activation == "sigmoid":
        a, activation_cache = sigmoid(z)
    elif activation == "softmax":
        a, activation_cache = softmax(z)
    cache = linear_cache, activation_cache
    return a, cache


def forward(x, params, activation, last_activation, num_of_layers):
    a = x
    caches = []
    for i in range(1, num_of_layers):
        if i != num_of_layers - 1:
            a, cache = single_forward(a, params['w' + str(i)], params['b' + str(i)], activation)
        else:
            a, cache = single_forward(a, params['w' + str(i)], params['b' + str(i)], last_activation)
        caches.append(cache)
    return a, caches


def calc_loss(y, a, m, loss_function):
    if loss_function == "cross_entropy":
        return np.sum(-(y * np.log(a))) / m
    elif loss_function == "mean_square_error":
        return np.sum(np.square(y - a)) / (2 * m)


def loss_grad(y, a, m, loss_function):
    if loss_function == "cross_entropy":
        return -(y / a) / m
    elif loss_function == "mean_square_error":
        return (a - y) / m


def single_backward(da, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "sigmoid":
        dz = sigmoid_grad(da, activation_cache)
    elif activation == "relu":
        dz = relu_gradient(da, activation_cache)
    elif activation == "leaky_relu":
        dz = leaky_relu_gradient(da, activation_cache)
    elif activation == "softmax":
        dz = softmax_gradient(da, activation_cache)
    grads = linear_grad(dz, linear_cache)
    return grads['w'], grads['db'], grads['da']


def backward(y, a, m, caches, loss_function, activation, last_activation, num_of_layers):
    grads = {}
    dloss = loss_grad(y, a, m, loss_function)
    for i in reversed(range(1, num_of_layers)):
        if i == num_of_layers - 1:
            grads["dw" + str(i)], grads["db" + str(i)], grads["da" + str(i - 1)] = single_backward(dloss,
                                                                                                   caches[i - 1],
                                                                                                   last_activation)
        else:
            grads["dw" + str(i)], grads["db" + str(i)], grads["da" + str(i - 1)] = single_backward(grads["da" + str(i)],
                                                                                                   caches[i - 1],
                                                                                                   activation)
    return grads


def forward_and_backward(x, y, m, params, activation, loss_function, last_activation, num_of_layers):
    a, caches = forward(x, params, activation, last_activation, num_of_layers)
    loss = calc_loss(y, a, m, loss_function)
    grads = backward(y, a, m, caches, loss_function, activation, last_activation, num_of_layers)
    return grads, loss


def gradient_clip(grad, limit):
    if np.linalg.norm(grad) >= limit:
        grad = limit * (grad / np.linalg.norm(grad))
    return grad


def update_params(params, learning_rate, grads, num_of_layers):
    for i in range(1, num_of_layers):
        params["w" + str(i)] = params["w" + str(i)] - learning_rate * gradient_clip(grads["dw" + str(i)], 0.01)
        params["b" + str(i)] = params["b" + str(i)] - learning_rate * gradient_clip(grads["db" + str(i)], 0.01)
    return params


def predict(params, x, y, activation, last_activation, num_of_layers):
    a, _ = forward(x, params, activation, last_activation, num_of_layers)
    prediction = np.zeros(a.shape)
    prediction[a >= 0.9] = 1
    accuracy = np.mean(np.all(prediction == y_test, axis=1, keepdims=True)) * 100
    return accuracy


def make_gaussian_data(num_of_samples, negative_mean, negative_cov, positive_mean, positive_cov):
    negative_samples = np.random.multivariate_normal(mean=negative_mean, cov=negative_cov, size=num_of_samples)
    positive_samples = np.random.multivariate_normal(mean=positive_mean, cov=positive_cov, size=num_of_samples)

    x = np.vstack((negative_samples, positive_samples)).astype('float64')
    y = np.vstack((np.zeros((num_of_samples, np.array([1, 0]))),
                   np.ones((num_of_samples, np.array([0, 1]))))).astype('float64')

    plt.scatter(x[:, 0], x[:, 1], c=y[:, 0])
    plt.show()
    return x, y


def dnn_model(x_train, y_train, x_test, y_test,
              learning_rate, num_of_iterations,
              num_of_epochs, loss_function,
              activation, last_activation, dims_of_layers):
    dim = x_test.shape[1]
    params = init_params(dims_of_layers)
    losses = []
    num_of_layers = len(dims_of_layers)
    for epoch in range(num_of_epochs):
        for i, x in enumerate(x_train):
            for j in range(num_of_iterations):
                m = x.shape[0]
                y = y_train[i]
                grads, loss = forward_and_backward(x, y, m, params, activation, loss_function, last_activation,
                                                   num_of_layers)
                params = update_params(params, learning_rate, grads, num_of_layers)
                if j % 100 == 0:
                    losses.append(loss)
                    print(f"epoch: {epoch}, loss after iteration {j}:  {loss}")
                    # train_accuracy = predict(params, x, y)
                    # test_accuracy = predict(params, x_test, y_test)
                    # print(f"train accuracy: {train_accuracy}%")
                    # print(f"test accuracy: {test_accuracy}%")

    # train_accuracy = predict(params, x_train, y_train)
    # test_accuracy = predict(params, x_test, y_test)
    # print(f"final train accuracy: {train_accuracy}%")
    # print(f"final test accuracy: {test_accuracy}%")

    plt.figure()
    plt.plot(losses)
    plt.xlabel("num of iterations")
    plt.ylabel("loss")
    plt.title("deep neural network")
    plt.show()

    return params


x_train, y_train = make_gaussian_data(10000, negative_mean=[3.0, 1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                      positive_mean=[1.0, 3.0], positive_cov=[[2.0, 1.0], [1.0, 2.0]])
x_test, y_test = make_gaussian_data(1000, negative_mean=[2.0, -1.0], negative_cov=[[3.0, 1.0], [1.0, 3.0]],
                                    positive_mean=[-2.0, 2.0], positive_cov=[[4.0, 1.0], [1.0, 2.0]])
params = dnn_model(x_train, y_train, x_test, y_test,
                   learning_rate=0.001,
                   num_of_iterations=1000,
                   num_of_epochs=5,
                   loss_function="cross_entropy",
                   activation="relu",
                   last_activation="softmax",
                   dims_of_layers=[2, 4, 4, 1])
import numpy as np

import header as h
import matplotlib.pyplot as plt

number_of_data1 = 1000
number_of_data2 = 1000
number_of_data = number_of_data1 + number_of_data2
num_of_data_train = number_of_data
# dataset generation
mean1 = [3., 1.]
mean2 = [1., 3.]
cov1 = [[3.0, 1.0], [1.0, 3.0]]
cov2 = cov1

mean1_test = [4., 2.]
mean2_test = [0., 2.]
cov1_test = [[3.0, 1.0], [1.0, 3.0]]
cov2_test = cov1_test

num_of_features = 2
# train dataset
x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=number_of_data1)
x2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=number_of_data2)
x_train = np.vstack((x1, x2))

x1_test = np.random.multivariate_normal(mean=mean1_test, cov=cov1_test, size=number_of_data1)
x2_test = np.random.multivariate_normal(mean=mean2_test, cov=cov2_test, size=number_of_data2)
x_test = np.vstack((x1_test, x2_test))

y1 = np.zeros((number_of_data1, 1), dtype='float32')
y2 = np.ones((number_of_data2, 1), dtype='float32')
y_train = np.vstack((y1, y2))

y1_test = np.zeros((number_of_data1, 1), dtype='float32')
y2_test = np.ones((number_of_data2, 1), dtype='float32')
y_test = np.vstack((y1_test, y2_test))

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
h.params = {}
for i in range(1, num_of_layers):
    h.params["w" + str(i)] = np.random.rand(layers[i - 1], layers[i]) * 0.001
    h.params["b" + str(i)] = np.zeros((1, layers[i]))


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



h.grads = {}
h.a = {}
h.z = {}


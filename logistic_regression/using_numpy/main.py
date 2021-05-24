"""
Logistic regression model
In machine learning, vectors are often represented as column vectors.
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


number_of_data1 = 1000
number_of_data2 = 1000
number_of_data = number_of_data1 + number_of_data2

# dataset generation
mean1 = [3., 1.]
mean2 = [1., 3.]
cov1 = [[3.0, 1.0], [1.0, 3.0]]
cov2 = cov1

mean1_test = [4., 2.]
mean2_test = [0., 2.]
cov1_test = [[3.0, 1.0], [1.0, 3.0]]
cov2_test = cov1_test

# train dataset
x1 = np.random.multivariate_normal(mean=mean1, cov=cov1, size=number_of_data1)
x2 = np.random.multivariate_normal(mean=mean2, cov=cov2, size=number_of_data2)
X = np.vstack((x1, x2))

x1_test = np.random.multivariate_normal(mean=mean1_test, cov=cov1_test, size=number_of_data1)
x2_test = np.random.multivariate_normal(mean=mean2_test, cov=cov2_test, size=number_of_data2)
X_test = np.vstack((x1_test, x2_test))

y1 = np.zeros((number_of_data1, 1), dtype='float32')
y2 = np.ones((number_of_data2, 1), dtype='float32')
Y = np.vstack((y1, y2))
p = Y

y1_test = np.zeros((number_of_data1, 1), dtype='float32')
y2_test = np.ones((number_of_data2, 1), dtype='float32')
Y_test = np.vstack((y1_test, y2_test))

plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_test[:, 0])
plt.show()
dim = np.size(X, 1)

# initializing parameters
w = np.random.rand(dim, 1) * 0.001
b = np.zeros(shape=(number_of_data, 1))

Y_hat = np.zeros(shape=[number_of_data, 1])
p_hat = np.zeros(shape=[number_of_data, 1])
grad_L = {"dw": 0.0, "db": 0.0}
alpha = 0.001
lamb = 0.05


def forward_propagation(xx):
    global Y_hat, p_hat
    Y_hat = np.matmul(xx, w) + b
    p_hat = sigmoid(Y_hat)
    return None


def back_propagation():
    global grad_L
    m = number_of_data
    round_L_round_p_hat = -(p / p_hat - (1. - p) / (1. - p_hat)) / m
    round_p_hat_round_Y_hat = sigmoid(Y) * (1. - sigmoid(Y))
    temp = round_L_round_p_hat * round_p_hat_round_Y_hat

    grad_L = {"dw": np.matmul(X.transpose(), temp),
              "db": np.mean(temp)}
    return None


def update_params():
    global w, b, lamb
    # w = w - alpha * (grad_L["dw"] + lamb * w)
    w = w * (1 - alpha * lamb) - alpha * grad_L["dw"]
    b = b - alpha * grad_L["db"]
    return None


def calc_loss():
    return np.sum(-(p * np.log(p_hat) + (1 - p) * np.log(1 - p_hat))) / number_of_data


def calc_accuracy(xx, yy):
    forward_propagation(xx)
    prediction = np.zeros(p_hat.shape)
    prediction[p_hat >= 0.9] = 1
    accuracy1 = (1.0 - np.mean(np.abs(prediction - yy))) * 100
    return accuracy1


def shuffle_data():
    random.Random(1000).shuffle(X)
    random.Random(1000).shuffle(Y)
    return None


# main calculation
number_of_epochs = 10
number_of_iterations = 3000
total_loss = []
for epoch in range(number_of_epochs):
    shuffle_data()  # this effect is so dramatic???
    for i in range(number_of_iterations):
        forward_propagation(X)  # loss function calculation
        back_propagation()  # gradient calculation
        update_params()
        if i % 100 == 0:
            total_loss.append(calc_loss())
            print(f'loss after iteration {i}: {calc_loss()}')
            print(f'train accuracy: {calc_accuracy(X, Y)}')
            print(f'test accuracy: {calc_accuracy(X_test, Y_test)}')

plt.figure()
plt.yscale('log')
plt.plot(total_loss)
plt.xlabel(f"num of iter * {int(1e+2)}")
plt.ylabel("loss")
plt.title("logistic regression")
plt.show()

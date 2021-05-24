import numpy as np

from header import *
import matplotlib.pyplot as plt

num_of_data_train = 1000
num_of_data_test = int(num_of_data_train * 0.2)
#
num_of_features = 30
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
np.random.shuffle(x_test)
np.random.shuffle(y_test)
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


grads = {}
a = {}
z = {}
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
                z[ilayer] = np.matmul(x, w) + b
                if ilayer == num_of_layers - 1:
                    a[ilayer] = activation_function(z[ilayer], 'sigmoid')
                else:
                    a[ilayer] = activation_function(z[ilayer], activation_func)
                x = a[ilayer]

            for ilayer in range(num_of_layers - 1, 0, -1):
                if ilayer == num_of_layers - 1:
                    dLda = grad_loss(a[ilayer], y_train, num_of_data_train, loss_func)
                    dadz = grad_actiavtion_function(z[ilayer], 'sigmoid')
                    # temp = linear_grad(dLda, dadz)
                    temp = np.matmul(dadz.transpose(), dLda)  # element by element operation
                    dzdw = a[ilayer - 1]
                    grads["dw" + str(ilayer)] = np.matmul(dzdw.transpose(), temp)
                    grads["db" + str(ilayer)] = np.mean(temp)
                else:
                    dzda = params["w" + str(ilayer)]
                    dadz = grad_actiavtion_function(z[ilayer], activation_func)
                    temp = temp * dzda
                    temp = temp * dadz
                    if ilayer == 1:
                        dzdw = x_train
                    else:
                        dzdw = a[ilayer]
                grads["dw" + str(ilayer)] = np.matmul(dzdw.transpose(), temp)
                grads["db" + str(ilayer)] = np.mean(temp)
            for ilayer in range(1, num_of_layers):
                params["w" + str(ilayer)] = params["w" + str(ilayer)] - alpha * grads["dw" + str(ilayer)]
                params["b" + str(ilayer)] = params["b" + str(ilayer)] - alpha * grads["db" + str(ilayer)]

            if j % 100 == 0:
                loss = calc_loss(y_train, a, num_of_data_train, loss_func)
            total_loss = total_loss.append(loss)
            print(f"epoch: {i}, iteration: {j} loss: {loss}")

    plt.figure()
    plt.plot(total_loss)
    plt.xlabel("num of iterations")
    plt.ylabel("loss")
    plt.title("DNN")
    plt.show()

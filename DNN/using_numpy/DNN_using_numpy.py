from header import *
import matplotlib.pyplot as plt

num_of_data_train = 1000
num_of_data_test = int(num_of_data_train * 0.2)
#
num_of_features = 20
#
x_train = np.random.randn(num_of_data_train, num_of_features)
x_test = np.random.randn(num_of_data_test, num_of_features)
y_train = np.zeros(shape=(num_of_data_train,1))
y_test = np.zeros(shape=(num_of_data_test,1))
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
layers = [num_of_features, 16, 16, 1]
num_of_layers = len(layers)
#
alpha = 0.001
lamb = 0.001
#
params = {}
input()
for i in range(1, num_of_layers):
    params["w" + str(i)] = np.random.rand(layers[i-1], layers[i]) * 0.001
    params["b" + str(i)] = np.zeros((1, layers[i]))


def forward(x_start,activation_function):
    for ilayer in range(1, num_of_layers):
        w = params["w"+str(ilayer)]
        b = params["b"+str(ilayer)]
        if ilayer == 1:
            x = x_start
            z = np.matmul(x, w) + b
            if activation_function == "sigmoid":
                a = sigmoid(z)
            elif activation_function == "relu":
                a = relu(z)
            elif activation_function == "leaky_relu":
                a = leaky_relu(z)
        elif ilayer == num_of_layers-1:
            x = a
            z = np.matmul(x, w) + b
            a = softmax(z)
        else:
            x = a
            z = np.matmul(x,w) + b
            if activation_function == "sigmoid":
                a = sigmoid(z)
            elif activation_function == "relu":
                a = relu(z)
            elif activation_function == "leaky_relu":
                a = leaky_relu(z)
    return a


def backward(x_start,y,a,num_of_data, activation_function):
    grads = {}
    for ilayer in range(num_of_layers-1, 0, -1):
        A = grad_loss(a,y,num_of_data,'cross_entropy')
        if ilayer == num_of_layers-1:
            B = grad_softmax(z)

            grads["dw"+str(ilayer)] = grad_loss()




def grad_loss(a, y, m, loss_function):
    if loss_function == 'cross_entropy':
        return - y * np.log(a) / m
    elif loss_function == 'mse':
        return (a - y) / m


if __name__ == "__main__":
    epochs = 100
    num_of_iterations = 1000
    for i in range(epochs):
        for j in range(num_of_iterations):
            pass

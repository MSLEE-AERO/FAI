from header import *

num_of_data_train = 1000
num_of_data_test = int(num_of_data_train * 0.2)
#
num_of_features = 20
#
x_train = np.zeros(shape=[num_of_data_train, num_of_features])
x_test = np.zeros(shape=[num_of_data_test, num_of_features])
y_train = np.zeros(shape=num_of_data_train)
y_test = np.zeros(shape=num_of_data_test)
#
for m in range(num_of_data_train):
    if m > 0.5 * num_of_data_train:
        y_train[m] = 1
        for n in range(num_of_features):
            x_train[m, n] = 1
np.random.shuffle(x_train)
np.random.shuffle(y_train)
#
for m in range(num_of_data_test):
    if m > 0.5 * num_of_data_test:
        y_test[m] = 1
        for n in range(num_of_features):
            x_test[m, n] = 1
np.random.shuffle(x_test)
np.random.shuffle(y_test)
#
layers = [num_of_features, 16, 16, 4]
num_of_layers = len(layers)
#
alpha = 0.001
lamb = 0.001
#
params = {}
for i in range(num_of_layers - 1):
    params["w" + str(i)] = np.random.rand(layers[i], layers[i + 1]) * 0.001
    params["b" + str(i)] = np.zeros((1, layers[i + 1]))


def forward(x, w, b, activation_function):
    z = np.matmul(x, w) + b
    if activation_function == "sigmoid":
        a = sigmoid(z)
    elif activation_function == "relu":
        a = relu(z)
    elif activation_function == "leaky_relu":
        a = leaky_relu(z)
    elif activation_function == "softmax":
        a = softmax(z)
    else:
        print("activation_function error")
        return -1
    return a


def grad_loss(a,y,m,loss_function):
    if loss_function == 'cross_entropy':
        return - y*np.log(a) / m
    elif loss_function == 'mse':
        return

if __name__ == "__main__":
    epochs = 100
    num_of_iterations = 1000
    for i in range(epochs):
        for j in range(num_of_iterations):






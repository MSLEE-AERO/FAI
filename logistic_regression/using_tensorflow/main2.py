from functions import *
import matplotlib.pyplot as plt
from tensorflow.math import log

#
num_of_data1 = 100
num_of_data2 = 100
num_of_data = num_of_data1 + num_of_data2
#
num_of_data1_train = int(num_of_data1 * .2)
num_of_data2_train = int(num_of_data2 * .2)
num_of_data_train = num_of_data1_train + num_of_data2_train
#
num_of_features = 2
#
x1 = tf.random.normal(shape=[num_of_data1, num_of_features], mean=[1, 3], stddev=1.0)
x2 = tf.random.normal(shape=[num_of_data2, num_of_features], mean=[3, 1], stddev=1.0)
X = tf.experimental.numpy.vstack((x1, x2))
#
y1 = tf.zeros(shape=[num_of_data1, 1])
y2 = tf.ones(shape=[num_of_data2, 1])
Y = tf.experimental.numpy.vstack((y1, y2))
p = Y
#
x1_train = tf.random.normal(shape=[num_of_data1_train, num_of_features], mean=[2, 3], stddev=1.0)
x2_train = tf.random.normal(shape=[num_of_data2_train, num_of_features], mean=[5, 1], stddev=1.0)
X_train = tf.experimental.numpy.vstack((x1_train, x2_train))
#
y1_train = tf.zeros(shape=[num_of_data1_train, 1])
y2_train = tf.ones(shape=[num_of_data2_train, 1])
Y_train = tf.experimental.numpy.vstack((y1_train, y2_train))
#
plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])
plt.scatter(X_train[:, 0], X_train[:, 1], c=Y_train[:, 0])
plt.show()
#
#
w = tf.Variable(tf.random.normal(shape=(num_of_features, 1)))
b = tf.Variable(tf.random.normal(shape=(num_of_data, 1)))
#
Y_hat = tf.zeros(shape=(num_of_data, 1))
p_hat = tf.zeros(shape=(num_of_data, 1))
grad_L = []
#
# initialize hyper parameter
alpha = 0.001
lamb = 0.001


def forward_propagation(xx):
    global Y_hat, p_hat
    Y_hat = matmul(xx, w) + b
    p_hat = sigmoid(Y_hat)
    return None


def update_params():
    global w, b
    w = w - alpha * (grad_L[0] + lamb * w)
    b = b - alpha * grad_L[1]
    return None


def calc_loss():
    return sum(
        -(p * ln(p_hat) + (1. - p) * ln(1. - p_hat)) / num_of_data
    )


def calc_accuracy(xx, yy):
    forward_propagation(xx)
    prediction = tf.zeros(p_hat.shape)
    prediction[p_hat >= 0.9] = 1
    accuracy1 = (1.0 - tf.math.reduce_mean(tf.abs(prediction - yy))) * 100
    return accuracy1


def shuffle_data():
    tf.random.shuffle(seed=1000,value=X)
    tf.random.shuffle(seed=1000,value=Y)
    return None


#@tf.function
def calc_gradient():
    global grad_L
    with tf.GradientTape() as tape:
        forward_propagation(X)
        loss = calc_loss()
        grad_L = tape.gradient(loss, [w, b])
    return None


#
#
if __name__ == '__main__':
    num_of_epochs = 100
    num_of_iterations = 10000
    total_loss = []
    for epoch in range(num_of_data):
        shuffle_data()
        for i in range(num_of_iterations):
            calc_gradient()
            update_params()
            if i % 100 == 0:
                print(f'epoch is {epoch}, iter# is {i}')
                total_loss.append(calc_loss())
                print(f'loss at iteration {i}: {calc_loss()}')
                print(f'test accuracy: {calc_accuracy(X_train, Y_train)}')

    plt.figure()
    plt.yscale('log')
    plt.plot(f"sum of iter * {int(1e+2)}")
    plt.ylabel("loss")
    plt.title("logistic regression")
    plt.show()


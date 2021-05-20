from functions import *
import matplotlib.pyplot as plt
import numpy as np

# global variables
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
b = tf.Variable(tf.zeros(shape=(1,)))
#
#
# initialize hyper parameter
alpha = 0.001
lamb = 0.001


def forward_propagation(xx, ww, bb):
    y_hat = tf.linalg.matmul(xx, ww) + bb
    p_hat = tf.math.sigmoid(y_hat)
    return p_hat


def calc_loss(p, p_hat, m):
    loss_local = tf.math.reduce_sum(
        -(p * tf.math.log(p_hat) + (1. - p) * tf.math.log(1. - p_hat))) / m
    return loss_local


@tf.function
def calc_gradient(x, y, m, ww, bb):
    with tf.GradientTape() as tape:
        p_hat = forward_propagation(x, ww, bb)
        loss_local = calc_loss(y, p_hat, m)
        grad = tape.gradient(target=loss_local, sources=[ww, bb])
    return grad, loss_local


# return None vs return grad_L, what is difference??


def update_params(grad, w, b):
    w_new = w.assign_sub(alpha * grad[0])
    b_new = b.assign_sub(alpha * grad[1])
    return w_new, b_new


def calc_accuracy(w, b, xx, yy):
    p_hat = forward_propagation(xx, w, b)
    prediction = np.zeros(p_hat.shape)
    prediction[p_hat.numpy() >= 0.9] = 1.
    accuracy1 = (1.0 - tf.math.reduce_mean(tf.abs(prediction - yy))) * 100
    return accuracy1


def shuffle_data():
    tf.random.shuffle(seed=1000, value=X)
    tf.random.shuffle(seed=1000, value=Y)
    return None


if __name__ == '__main__':
    num_of_epochs = 100
    num_of_iterations = 10000
    total_loss = []
    for epoch in range(num_of_epochs):
        # shuffle_data()
        for i in range(num_of_iterations):
            grad_L, loss = calc_gradient(X, Y, num_of_data, w, b)
            w, b = update_params(grad_L, w, b)
            if i % 100 == 0:
                print(f'epoch is {epoch + 1}, iter# is {i + 1}')
                total_loss.append(loss)
                print(f'loss at iteration {i + 1}: {loss}')
                print(f'test accuracy: {calc_accuracy(w, b, X_train, Y_train)}')

    plt.figure()
    plt.yscale('log')
    plt.plot(f"sum of iter * {int(1e+2)}")
    plt.ylabel("loss")
    plt.title("logistic regression")
    plt.show()

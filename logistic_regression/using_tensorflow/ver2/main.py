import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def init_params(dim):
    w = tf.Variable(tf.random.normal(shape=(dim,1)))
    b = tf.Variable(tf.zeros(shape=(1,)))
    return w, b


def forward(x, w, b):
    z = tf.linalg.matmul(x, w) + b
    a = tf.math.sigmoid(z)
    return a


def calc_loss(y, a, m, loss_function, w):
    if loss_function == "cross_entropy":
        return tf.math.reduce_sum(
            -(y * tf.math.log(a) + (1 - y)*tf.math.log(1-a))
        ) / m
    elif loss_function == "MSE":
        return tf.math.reduce_sum(tf.math.square(y - a))/m
    else:
        print('loss function input error')
        return -1


@tf.function
def forward_and_backward(x, y, m, w, b, loss_function):
    with tf.GradientTape() as tape:
        a = forward(x, w, b)
        loss = calc_loss(y, a, m, loss_function, w)
        grads = tape.gradient(loss, [w, b])
    return grads, loss


def update_params(w, b, learning_rate, grads):
    w.assign_sub(learning_rate * grads[0])
    b.assign_sub(learning_rate * grads[1])
    return w, b


def predict(w, b, x, y):
    a = forward(x, w, b)
    prediction = np.zeros(a.shape)
    prediction[a >= 0.9] = 1
    accuracy = (1 - np.mean(np.abs(prediction - y))) * 100
    return accuracy


def make_data(num_of_data1, num_of_data2, num_of_features,mean1,mean2,std):
    ndata1 = num_of_data1
    ndata2 = num_of_data2
    nfeatures = num_of_features

    x1 = tf.random.normal(shape=[ndata1,nfeatures],mean=mean1,stddev=std)
    x2 = tf.random.normal(shape=[ndata2, nfeatures], mean=mean2, stddev=std)
    x = tf.experimental.numpy.vstack((x1,x2))

    y1 = tf.zeros(shape=[ndata1,1])
    y2 = tf.ones(shape=[ndata2,1])
    y = tf.experimental.numpy.vstack((y1,y2))

    plt.scatter(x[:,0],x[:,1],c=y[:,0])
    plt.show()
    return x, y


if __name__ == "__main__":
    X_train, Y_train = make_data(num_of_data1=1000,num_of_data2=1000,num_of_features=2,mean1=[1,3],mean2=[3,1],std=1.0)
    X_test, Y_test = make_data(num_of_data1=200,num_of_data2=200,num_of_features=2,mean1=[2,3],mean2=[5,1],std=1.0)
    dataset = tf.data.Dataset.from_tensor_slices((X_train,Y_train))
    epochs = 100
    niter = 1000
    loss_function = 'cross_entropy'
    dim = X_train.shape[1]
    w, b = init_params(dim)
    total_loss = []
    for epoch in range(epochs):
        batch_dataset = dataset.shuffle(buffer_size=1000).batch(1000)
        for i, (x, y) in enumerate(batch_dataset):
            for j in range(niter):
                m = x.shape[0]
                grads, loss = forward_and_backward(X_train,Y_train,m,w, b,loss_function)
                w, b = update_params(w, b, learning_rate=0.001,grads=grads)
                if j % 100 == 0:
                    total_loss.append(loss)
                    print(f"epoch: {epoch}, # iteration: {j}: {loss}")
                    train_accuracy = predict(w, b, x, y)
                    test_accuracy = predict(w, b, X_test, Y_test)
                    print(f"train accuracy: {train_accuracy}%")
                    print(f"test accuracy: {test_accuracy}%")
    train_accuracy = predict(w, b, X_train, Y_train)
    test_accuracy = predict(w, b, X_test, Y_test)
    print(f"final train accuracy: {train_accuracy}")
    print(f"final test accuracy: {test_accuracy}")

    plt.figure()
    plt.plot(total_loss)
    plt.xlabel("# iterations")
    plt.ylabel("loss")
    plt.title("logistic regression")




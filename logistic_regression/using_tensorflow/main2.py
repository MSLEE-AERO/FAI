import tensorflow as tf
import matplotlib.pyplot as plt
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
x1 = tf.random.normal(shape=[num_of_data1,num_of_features],mean=[1,3],stddev=1.0)
x2 = tf.random.normal(shape=[num_of_data2,num_of_features],mean=[3,1],stddev=1.0)
X = tf.experimental.numpy.vstack((x1,x2))
#
y1 = tf.zeros(shape=[num_of_data1,1])
y2 = tf.ones(shape=[num_of_data2,1])
Y = tf.experimental.numpy.vstack((y1,y2))
#
x1_train = tf.random.normal(shape=[num_of_data1_train,num_of_features],mean=[2,3],stddev=1.0)
x2_train = tf.random.normal(shape=[num_of_data2_train,num_of_features],mean=[5,1],stddev=1.0)
X_train = tf.experimental.numpy.vstack((x1_train,x2_train))
#
y1_train = tf.zeros(shape=[num_of_data1_train,1])
y2_train = tf.ones(shape=[num_of_data2_train,1])
Y_train = tf.experimental.numpy.vstack((y1_train,y2_train))
#
plt.scatter(X[:,0],X[:,1], c=Y[:,0])
plt.scatter(X_train[:,0],X_train[:,1], c=Y_train[:,0])
plt.show()

class logisticModel():
    w = tf.random.normal(shape=[])

    def init_params(self, features,):



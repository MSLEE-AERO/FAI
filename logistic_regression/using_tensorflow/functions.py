import tensorflow as tf
#
#
def ln(x):
    return tf.math.log(x)
#
#
def sigmoid(x):
    return tf.math.sigmoid(x)
#
#
def matmul(A,B):
    return tf.linalg.matmul(A,B)
#
#
def sum(x):
    return tf.experimental.numpy.sum(x)
#
#

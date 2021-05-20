import tensorflow as tf

def mat1(a,b):
    return a * b

@tf.function
def mat(a,b):
    return a * b


a = tf.random.normal(shape=(2,2))
b = tf.random.normal(shape=(2,2))

print(mat(a,b))
print(mat1(a,b))
print(tf.matmul(a,b))
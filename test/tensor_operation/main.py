import tensorflow as tf

a = tf.Variable(1)

b = tf.Variable(2)

c = a + b
d = 0.1

x = tf.random.normal(shape=(2,3))
y = tf.random.normal(shape=(2,3))

@tf.function
def calc_gradient():
    with tf.GradientTape() as tape:
        grad = tape.gradient()




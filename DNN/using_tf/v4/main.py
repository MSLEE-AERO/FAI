import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i])
    plt.xlabel(y_train[i])
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))


class CustomDense1(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomDense1, self).__init__()
        self.w = tf.Variable(initial_value=tf.random.normal(shape=(input_dim, units), mean=0, stddev=1.),
                             trainable=True)
        self.b = tf.Variable(initial_value=tf.zeros(shape=(units,)),
                             trainable=True)
        self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        return self.activation(tf.linalg.matmul(inputs, self.w) + self.b)


custom_dense1 = CustomDense1(input_dim=2, units=1, activation='relu')

assert custom_dense1.weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.trainable_weights == [custom_dense1.w, custom_dense1.b]
assert custom_dense1.non_trainable_weights == []


class CustomDropout(tf.keras.layers.Layer):
    def __init__(self, rate):
        super(CustomDropout, self).__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs


class CustomCombinationLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, units, activation):
        super(CustomCombinationLayer, self).__init__()
        self.custom_dense1 = CustomDense1(input_dim, units, activation)
        self.custom_dropout = CustomDropout(0.5)

    def call(self, inputs, training=None):
        a = self.custom_dense1(inputs)
        return self.custom_dropout(a, training=training)


class CustomModel(tf.keras.models.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.hidden_layer1 = CustomCombinationLayer(input_dim=784, units=16, activation='relu')
        self.hidden_layer2 = CustomCombinationLayer(input_dim=16, units=16, activation='relu')
        self.hidden_layer3 = CustomCombinationLayer(input_dim=16, units=16, activation='relu')
        self.output_layer = CustomCombinationLayer(input_dim=16, units=10, activation='softmax')

    def call(self, inputs, training=True):
        hidden1 = self.hidden_layer1(inputs, training=training)
        hidden2 = self.hidden_layer1(hidden1, training=training)
        hidden3 = self.hidden_layer1(hidden2, training=training)
        return self.output_layer(hidden3, training=training)


model = CustomModel()

optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()
accuracy = tf.keras.metrics.CategoricalAccuracy()


@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy.update_state(y, logits)

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = model(x, training=False)
    accuracy.update_state(y, logits)


for epoch in range(5):
    print('%d epoch' % (epoch + 1))
    for i, (x, y) in enumerate(train_data):
        loss_value = train_step(x, y)
        if i % 100 == 0:
            print('%d step / loss value: %f / accuracy: $f' % (i, float(loss_value), float(accuracy.result())))

    for step, (x, y) in enumerate(test_data):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))

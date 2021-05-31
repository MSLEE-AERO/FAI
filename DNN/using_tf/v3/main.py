import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

input_layer = tf.keras.Input(shape=(28, 28))
input_flatten = tf.keras.layers.Flatten(input_shape=(28, 28))(input_layer)
hidden1 = tf.keras.layers.Dense(units=16, activation='relu')(input_flatten)
hidden2 = tf.keras.layers.Dense(units=16, activation='relu')(hidden1)
## question!
#hidden3 = tf.keras.layers.Dense(units=16, activation='relu')(hidden1 + hidden2)
hidden3 = tf.keras.layers.Dense(units=16, activation='relu')(hidden2)
output_layer = tf.keras.layers.Dense(units=10, activation='softmax')(hidden3)
model = tf.keras.Model(input_layer, output_layer)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)

model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

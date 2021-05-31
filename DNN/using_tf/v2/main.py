import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train[:].astype('float32') / 255
x_test = x_test[:].astype('float32') / 255

y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=5)

model.evaluate(x_train, y_train)
model.evaluate(x_test, y_test)

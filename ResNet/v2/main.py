# padding: valid: don't use zero-padding so loss of image data exist
# padding same: use zero-padding
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import numpy as np

K.set_image_data_format('channels_last')

# 1. create data
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size

)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    #plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds.shuffle(buffer_size=1000)
#train_ds = train_ds.cache().shuffle(buffer_size=1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
print(train_ds)
for i, (x, y) in enumerate(train_ds):
    print('i')
    print(i)
    print('x')
    print(x)
    print('y')
    print(y)
    break
input()

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(map_func=lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

print(np.min(first_image), np.max(first_image))

num_classes = 5


# 2. create layer
class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self,
                 activation,
                 filter_stride):
        super().__init__()
        self.activation = tf.keras.activations.get(activation)
        self.filter_stride = filter_stride

    def call(self, inputs, *args, **kwargs):
        if 'f' and 'filters' and 'stage' and 'blocks' in kwargs.keys():
            f = kwargs.get('f')
            filters = kwargs.get('filters')
            stage = kwargs.get('stage')
            blocks = kwargs.get('blocks')

            F1, F2, F3 = filters
            X = inputs

            for i in range(blocks):
                X_shortcut = tf.keras.layers.Conv2D(filters=F3,
                                                    kernel_size=(1, 1),
                                                    strides=self.filter_stride)(X)
                X = tf.keras.layers.Conv2D(filters=F1,
                                           kernel_size=(1, 1),
                                           strides=self.filter_stride,
                                           padding='valid')(X)
                X = tf.keras.layers.BatchNormalization(axis=3)(X)
                X = tf.keras.layers.Activation(self.activation)(X)

                X = tf.keras.layers.Conv2D(filters=F2,
                                           kernel_size=(f, f),
                                           strides=self.filter_stride,
                                           padding='same')(X)
                X = tf.keras.layers.BatchNormalization(axis=3)(X)
                X = tf.keras.layers.Activation(self.activation)(X)

                X = tf.keras.layers.Conv2D(filters=F3,
                                           kernel_size=(1, 1),
                                           strides=self.filter_stride,
                                           padding='valid')(X)
                X = tf.keras.layers.BatchNormalization(axis=3)(X)
                X = tf.keras.layers.Add()([X, X_shortcut])
                X = tf.keras.layers.Activation(self.activation)(X)

            return X
        else:
            raise ValueError('Invalid parameters')


# 3. build model (ResNet 50)
class ResNet50(tf.keras.models.Model):
    def __init__(self, name):
        super().__init__(name=name)

    def call(self, inputs, training=True, mask=None):
        # ResNet 50 layer
        # conv1 : 7 * 7, 64, stride 2

        X_input = tf.keras.layers.Input(shape=inputs.shape, tensor=inputs)
        X = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(X_input)

        X = tf.keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

        residualBlock = IdentityBlock(
            activation='relu',
            filter_stride=(1, 1)
        )
        X = residualBlock(X, f=3, filters=[64, 64, 256], blocks=3, stage=1)
        X = residualBlock(X, f=3, filters=[128, 128, 512], blocks=4, stage=2)
        X = residualBlock(X, f=3, filters=[256, 256, 1024], blocks=6, stage=3)
        X = residualBlock(X, f=3, filters=[512, 512, 2048], blocks=3, stage=4)

        X = tf.keras.layers.GlobalAveragePooling2D(name='avg-pooling')(X)
        # X = tf.keras.layers.AveragePooling2D(pool_size=(3, 3))(X)
        X = tf.keras.layers.Dense(units=num_classes, activation='softmax')(X)

        return X


model = ResNet50(name='Custom_Resnet50')
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
accuracy = tf.keras.metrics.CategoricalAccuracy()



# 4. model training
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        print('in train_step')
        print(y)

        input()
        loss_value = loss(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy.update_state(y, logits)
    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = model(x)
    accuracy.update_state(y, logits)


# 5. evaluate model
for epoch in range(1):
    print('%d th epoch' % (epoch + 1))
    for index, (x, y) in enumerate(train_ds):
        print('before train_step')
        print(y)
        loss_value = train_step(x, y)
        if index & 100 == 0:
            print('%d index / loss_value: %f / accuracy: %f' % (index, float(loss_value), float(accuracy.result())))
    for step, (x, y) in enumerate(val_ds):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))

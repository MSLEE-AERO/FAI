import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import numpy as np

K.set_image_data_format('channels_last')

# 1. 데이터
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
    plt.show()

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixels values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

num_classes = 5


# 2. 레이어 생성
class ConvolutionBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, prev_a_channel, num_of_filters, activation, pad_size, filter_stride):
        super(ConvolutionBlock, self).__init__()
        self.filter_w = tf.Variable(
            initial_value=tf.random.normal(shape=(filter_size, filter_size, prev_a_channel, num_of_filters),
                                           mean=0., stddev=1.),
            trainable=True)
        self.filter_b = tf.Variable(
            initial_value=tf.zeros(shape=(1, 1, 1, num_of_filters)),
            trainable=True)
        self.activation = tf.keras.activations.get(activation)
        self.pad_size = pad_size
        self.filter_stride = filter_stride

    def call(self, X, f, filters, stage, block, s=2):
        F1, F2, F3 = filters
        X_shortcut = X

        X = tf.keras.layers.Conv2D(F1, (1, 1), strides=(s, s))(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)

        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same')(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)

        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X_shortcut = tf.keras.layers.Conv2D(filters=F3, strides=s, padding='valid')(X_shortcut)
        X_shortcut = tf.keras.layers.BatchNormalization(axis=3)(X_shortcut)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)

        return X


class IdentityBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, prev_a_channel, num_of_filters, activation, pad_size, filter_stride):
        super(ConvolutionBlock, self).__init__()

    def call(self, X, f, filters, stage, block):
        F1, F2, F3 = filters
        X_shortcut = X

        X = tf.keras.layers.Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)

        X = tf.keras.layers.Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same")(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation('relu')(X)

        X = tf.keras.layers.Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid')(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Add()([X, X_shortcut])
        X = tf.keras.layers.Activation('relu')(X)

        return X


# 3. 모델 빌드: resnet50
class Resnet50(tf.keras.models.Model):
    def __init__(self):
        super(Resnet50, self).__init__()

    def call(self, inputs, training=True):
        # CONV2D > BATCHNORM > RELU > MAXPOOL
        # > ConvolutionBlock > IDBLOCK * 2
        # > ConvolutionBlock > IDBLOCK * 3
        # > ConvolutionBlock > IDBLOCK * 5
        # > ConvolutionBlock > IDBLOCK * 2
        # > AVGPOOL > FLATTEN > FC

        X_input = tf.keras.layers.Input(input_shape)
        X = tf.keras.layers.ZeroPadding2D((3, 3))(X_input)

        X = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))(X)
        X = tf.keras.layers.BatchNormalization(axis=3)
        X = tf.keras.layers.Activation('relu')(X)
        X = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        X = ConvolutionBlock(X, f=3, filters=[64, 64, 256], stage=2, s=1)
        X = IdentityBlock(X, f=3, filters=[64, 64, 256], stage=2)(X)
        X = IdentityBlock(X, f=3, filters=[64, 64, 256], stage=2)(X)

        X = ConvolutionBlock(X, f=3, filters=[128, 128, 512], stage=3, s=2)
        X = IdentityBlock(X, f=3, filters=[128, 128, 512], stage=3)
        X = IdentityBlock(X, f=3, filters=[128, 128, 512], stage=3)
        X = IdentityBlock(X, f=3, filters=[128, 128, 512], stage=3)

        X = ConvolutionBlock(X, f=3, filters=[256, 256, 1024], stage=4, s=2)
        X = IdentityBlock(X, f=3, filters=[256, 256, 1024], stage=4)
        X = IdentityBlock(X, f=3, filters=[256, 256, 1024], stage=4)
        X = IdentityBlock(X, f=3, filters=[256, 256, 1024], stage=4)
        X = IdentityBlock(X, f=3, filters=[256, 256, 1024], stage=4)
        X = IdentityBlock(X, f=3, filters=[256, 256, 1024], stage=4)

        X = ConvolutionBlock(X, f=3, filters=[512, 512, 2048], stage=5, s=2)
        X = IdentityBlock(X, f=3, filters=[512, 512, 2048], stage=5)
        X = IdentityBlock(X, f=3, filters=[512, 512, 2048], stage=5)

        X = tf.keras.layers.AveragePooling2D((2, 2))(X)
        X = tf.keras.layers.Flatten()(X)
        X = tf.keras.layers.Dense(classes, activation='softmax')(X)

        return X


model = Resnet50()
optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.CategoricalCrossentropy()
accuracy = tf.keras.metrics.CategoricalAccuracy()
model.summary()


# 4. 모델 학습
@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss_value = loss(y, logits)
        gradients = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        accuracy.update_state(y, logits)

    return loss_value


def test_step(x, y):
    accuracy.reset_states()
    logits = model(x)
    accuracy.update_state(y, logits)


# 5. 모델 평가
for epoch in range(5):
    print('%d번째 epoch' % (epoch + 1))
    for index, (x, y) in enumerate(train_ds):
        loss_value = train_step(x, y)
        if index % 100 == 0:
            print('%d 단계 / loss_value: %f / accuracy: %f' % (index, float(loss_value), float(accuracy.result())))
    for step, (x, y) in enumerate(val_ds):
        test_step(x, y)
        print('test accuracy: %f' % (float(accuracy.result())))

# 6. 모델 배포
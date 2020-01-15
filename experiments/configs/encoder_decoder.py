import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense
from tensorflow.keras.layers import InputLayer, Reshape

DATASETS = {
    'mnist': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(50,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(50,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'cifar10': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(1024,)),  # latent dim
                Dense(4 * 4 * 128),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
            ]
        )
    }
}

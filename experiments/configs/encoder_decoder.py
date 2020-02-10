import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten
from tensorflow.keras.layers import InputLayer, Reshape
from tensorflow.keras.regularizers import l1, l2, l1_l2

VAE_ENC_DEC = {
    'mnist_v0': {
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
    'mnist_v1': {
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
                InputLayer(input_shape=(25,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'mnist_v2': {
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
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist_v0': {
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
    'fashion_mnist_v1': {
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
                InputLayer(input_shape=(25,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist_v2': {
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
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'cifar10_v0': {
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
    },
    'cifar10_v1': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5))
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v2': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5))
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(100,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    }
}


AE_ENC_DEC = {
    'mnist_v0': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(50)
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
    'mnist_v1': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(25)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(25,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'mnist_v2': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'mnist_v4': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Flatten(),
                Dense(1024, activation=tf.nn.relu),
                Dense(512, activation=tf.nn.relu),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(512, activation=tf.nn.relu),
                Dense(28 * 28 * 1, activation=tf.nn.sigmoid),
                Reshape(target_shape=(28, 28, 1)),
            ]
        )
    },
    'fashion_mnist_v0': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(50)
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
    'fashion_mnist_v1': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(25)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(25,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist_v2': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist_v3': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(7 * 7 * 32, activation=tf.nn.relu),
                Reshape(target_shape=(7, 7, 32)),
                Conv2DTranspose(64, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(32, 3, strides=2, padding='same', activation=tf.nn.relu),
                Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid')
            ]
        )
    },
    'fashion_mnist_v4': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Flatten(),
                Dense(1024, activation=tf.nn.relu),
                Dense(512, activation=tf.nn.relu),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(512, activation=tf.nn.relu),
                Dense(28 * 28 * 1, activation=tf.nn.sigmoid),
                Reshape(target_shape=(28, 28, 1)),
            ]
        )
    },
    'fashion_mnist_v5': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(28, 28, 1)),
                Flatten(),
                Dense(512, activation=tf.nn.relu),
                Dense(10)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(10,)),  # latent dim
                Dense(512, activation=tf.nn.relu),
                Dense(28 * 28 * 1, activation=tf.nn.sigmoid),
                Reshape(target_shape=(28, 28, 1)),
            ]
        )
    },
    'cifar10_v0': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
                Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
                Flatten(),
                Dense(1024)
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
    },
    'cifar10_v1': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v2': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(20)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(20,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v3': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(100)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(100,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v4': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v5': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v6': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid', kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v7': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Conv2D(32, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Flatten(),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(4 * 4 * 128, activation=tf.nn.relu),
                Reshape(target_shape=(4, 4, 128)),
                Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Conv2DTranspose(3, 4, strides=2, padding='same', activation=None, kernel_regularizer=l1(1e-5))
            ]
        )
    },
    'cifar10_v8': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Flatten(),
                Dense(512, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(512, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(32 * 32 * 3, activation=None),
                Reshape(target_shape=(32, 32, 3))
            ]
        )
    },
    'cifar10_v9': {
        'encoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(32, 32, 3)),
                Flatten(),
                Dense(1024, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(512, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(40)
            ]
        ),
        'decoder_net': tf.keras.Sequential(
            [
                InputLayer(input_shape=(40,)),  # latent dim
                Dense(512, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(1024, activation=tf.nn.relu, kernel_regularizer=l1(1e-5)),
                Dense(32 * 32 * 3, activation=None, kernel_regularizer=l1(1e-5)),
                Reshape(target_shape=(32, 32, 3))
            ]
        )
    }
}

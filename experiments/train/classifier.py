import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout
from tensorflow.keras.layers import Flatten, Input, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from alibi_detect.utils.saving import save_tf_model


def cnn_mnist(X_train):
    inputs = Input(shape=(X_train.shape[1:]))
    x = Conv2D(64, 2, padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(10, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_fashion_mnist(X_train):
    inputs = Input(shape=(X_train.shape[1:]))
    x = Conv2D(64, 2, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(10, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def cnn_cifar10(X_train):
    inputs = Input(shape=(X_train.shape[1:]))
    x = Conv2D(64, 2, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = Conv2D(32, 2, padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(.3)(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(.5)(x)
    logits = Dense(10, name='logits')(x)
    outputs = Activation('softmax', name='softmax')(logits)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def scale_by_instance(X: np.ndarray) -> np.ndarray:
    mean_ = X.mean(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    std_ = X.std(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    return (X - mean_) / std_


def run(dataset: str, epochs: int) -> None:

    # load and preprocess data
    if dataset == 'mnist':
        train, test = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        train, test = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        train, test = tf.keras.datasets.cifar10.load_data()
    X_train, y_train = train
    X_test, y_test = test
    shape = (-1,) + X_train.shape[1:]
    if len(shape) == 3:
        shape += (1,)
    X_train = X_train.reshape(shape).astype('float32')
    X_test = X_test.reshape(shape).astype('float32')
    y_train = y_train.astype('int64').reshape(-1,)
    y_test = y_test.astype('int64').reshape(-1,)
    n_cat = len(np.unique(y_train))
    y_train = to_categorical(y_train, n_cat)
    y_test = to_categorical(y_test, n_cat)

    # preprocess data
    if dataset in ['mnist', 'fashion_mnist']:
        X_train /= 255
        X_test /= 255
    elif dataset == 'cifar10':
        X_train = scale_by_instance(X_train)
        X_test = scale_by_instance(X_test)

    # define model
    if dataset == 'mnist':
        model = cnn_mnist(X_train)
    elif dataset == 'fashion_mnist':
        model = cnn_fashion_mnist(X_train)
    elif dataset == 'cifar10':
        model = cnn_cifar10(X_train)

    # create callback
    cp_path = '../models/clf/' + dataset + '_weak/ckpt/'
    cp_callback = ModelCheckpoint(filepath=cp_path, save_best_only=True, verbose=0)

    # train model
    model.fit(X_train,
              y_train,
              epochs=epochs,
              batch_size=128,
              shuffle=True,
              verbose=0,
              callbacks=[cp_callback],
              validation_data=(X_test, y_test))

    # save final model
    model_path = '../models/clf/' + dataset + '_weak/'
    save_tf_model(model, model_path)

    # evaluate model
    results = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    print('Test loss: {:.4f} -- accuracy: {:.4f}'.format(results[0], results[1]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train classifier.")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    run(args.dataset, args.epochs)

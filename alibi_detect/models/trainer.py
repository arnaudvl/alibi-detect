import logging
import math
import numpy as np
import os
import pickle
import tensorflow as tf
from typing import Tuple

logger = logging.getLogger(__name__)


HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3


def preprocess_batch(train_batch):
    train_batch_out = []
    for i in range(train_batch.shape[0]):
        train_batch_out.append(preprocess_image(train_batch[i], True))
    return tf.stack(train_batch_out)


def preprocess_image(image, is_training):
    """Preprocess a single image of layout [height, width, depth]."""
    # make sure type is right
    image = tf.cast(image, tf.float32)

    if is_training:
        # Resize the image to add four extra pixels on each side.
        image = tf.image.resize_with_crop_or_pad(image, HEIGHT + 8, WIDTH + 8)

        # Randomly crop a [HEIGHT, WIDTH] section of the image.
        image = tf.image.random_crop(image, [HEIGHT, WIDTH, NUM_CHANNELS])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)

    # Subtract off the mean and divide by the variance of the pixels.
    image = tf.image.per_image_standardization(image)
    return image


def save_ad_vae(model: tf.keras.Model, filepath: str) -> None:
    # create folder for model weights
    if not os.path.isdir(filepath):
        #logger.warning('Directory {} does not exist and is now created.'.format(filepath))
        os.mkdir(filepath)
    model_dir = os.path.join(filepath, 'model')
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # save encoder, decoder and vae weights
    model.encoder.encoder_net.save(os.path.join(model_dir, 'encoder_net.h5'))
    model.decoder.decoder_net.save(os.path.join(model_dir, 'decoder_net.h5'))
    model.save_weights(os.path.join(model_dir, 'vae.ckpt'))


def trainer(model: tf.keras.Model,
            loss_fn: tf.keras.losses,
            X_train: np.ndarray,
            y_train: np.ndarray = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss_fn_kwargs: dict = None,
            epochs: int = 20,
            batch_size: int = 64,
            buffer_size: int = 1024,
            verbose: bool = True,
            log_metric:  Tuple[str, "tf.keras.metrics"] = None,
            save_every: int = 1,
            save_path: str = None,
            preprocess: bool = False,
            callbacks: tf.keras.callbacks = None) -> None:  # TODO: incorporate callbacks + LR schedulers
    """
    Train TensorFlow model.

    Parameters
    ----------
    model
        Model to train.
    loss_fn
        Loss function used for training.
    X_train
        Training batch.
    y_train
        Training labels.
    optimizer
        Optimizer used for training.
    loss_fn_kwargs
        Kwargs for loss function.
    epochs
        Number of training epochs.
    batch_size
        Batch size used for training.
    buffer_size
        Maximum number of elements that will be buffered when prefetching.
    verbose
        Whether to print training progress.
    log_metric
        Additional metrics whose progress will be displayed if verbose equals True.
    callbacks
        Callbacks used during training.
    """
    # create directory if needed
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    # create dataset
    if y_train is None:  # unsupervised model without teacher forcing
        train_data = X_train
    else:
        train_data = (X_train, y_train)
    train_data = tf.data.Dataset.from_tensor_slices(train_data)
    train_data = train_data.shuffle(buffer_size=buffer_size).batch(batch_size)
    n_minibatch = int(np.ceil(X_train.shape[0] / batch_size))

    # init losses
    losses = {'kld': [], 'beta': [], 'recon': []}
    best_loss = 1e10

    # iterate over epochs
    for epoch in range(epochs):
        if verbose:
            pbar = tf.keras.utils.Progbar(n_minibatch, 1)

        # iterate over the batches of the dataset
        for step, train_batch in enumerate(train_data):

            if y_train is None:
                X_train_batch = train_batch
            else:
                X_train_batch, y_train_batch = train_batch

            if preprocess:
                X_train_batch = preprocess_batch(X_train_batch)

            with tf.GradientTape() as tape:
                preds = model(X_train_batch)

                if y_train is None:
                    ground_truth = X_train_batch
                else:
                    ground_truth = y_train_batch

                # compute loss
                if tf.is_tensor(preds):
                    args = [ground_truth, preds]
                else:
                    args = [ground_truth] + list(preds)

                if loss_fn_kwargs:
                    losses_minibatch = loss_fn(*args, **loss_fn_kwargs)
                    if len(losses_minibatch) == 2:
                        loss, loss_kld = losses_minibatch
                    else:
                        loss, loss_kld, loss_recon = losses_minibatch
                        losses['recon'].append(loss_recon.numpy())
                    losses['kld'].append(loss_kld.numpy())
                else:
                    loss = loss_fn(*args)

                if model.losses:  # additional model losses
                    losses['beta'].append(sum(model.losses).numpy())
                    #print(losses['beta'][-1])
                    loss += sum(model.losses)
                #    print('VAE loss: {}'.format(losses['beta'][-1]))
                #    print('total loss: {}'.format(loss.numpy()))

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            if verbose:
                loss_val = loss.numpy()
                if loss_val.shape:
                    if loss_val.shape[0] != batch_size:
                        if len(loss_val.shape) == 1:
                            shape = (batch_size - loss_val.shape[0], )
                        elif len(loss_val.shape) == 2:
                            shape = (batch_size - loss_val.shape[0], loss_val.shape[1])  # type: ignore
                        add_mean = np.ones(shape) * loss_val.mean()
                        loss_val = np.r_[loss_val, add_mean]
                pbar_values = [('loss', loss_val)]
                if log_metric is not None:
                    log_metric[1](ground_truth, preds)
                    pbar_values.append((log_metric[0], log_metric[1].result().numpy()))
                pbar.add(1, values=pbar_values)

        # print losses
        kld_mean = np.mean(np.array(losses['kld'][-step:])) if losses['kld'] else 0.
        beta_mean = np.mean(np.array(losses['beta'][-step:])) if losses['beta'] else 0.
        #beta_mean = 0. if math.isnan(beta_mean) else beta_mean
        recon_mean = np.mean(np.array(losses['recon'][-step:])) if losses['recon'] else 0.
        total_mean = kld_mean + recon_mean + beta_mean
        best_model = best_loss > total_mean
        print('Loss Epoch {}: Model KLD {:.4f} -- Latent KLD {:.4f} -- Recon {:.4f} -- Total {:.4f} -- Best {}'.format(
            epoch, kld_mean, beta_mean, recon_mean, total_mean, best_model))

        if epoch % save_every == 0:
            save_ad_vae(model, os.path.join(save_path, str(epoch)))

        if best_model:
            save_ad_vae(model, os.path.join(save_path, 'best'))
            best_loss = total_mean

    # save losses
    with open(os.path.join(save_path, 'losses.pickle'), 'wb') as f:
        pickle.dump(losses, f)

import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import kld
from tensorflow.keras.models import Model
from typing import Dict, Tuple
from alibi_detect.models.autoencoder import AE
from alibi_detect.models.trainer import trainer
from alibi_detect.models.losses import loss_adv_vae
from alibi_detect.base import BaseDetector, FitMixin, ThresholdMixin, adversarial_prediction_dict

logger = logging.getLogger(__name__)


class DenseHidden(tf.keras.Model):

    def __init__(self, model: tf.keras.Model, hidden_layer: int, output_dim: int, hidden_dim: int = None) -> None:
        """
        Dense layer that extracts the feature map of a hidden layer in a model and computes
        output probabilities over that layer.

        Parameters
        ----------
        model
        hidden_layer
        output_dim
        hidden_dim
        """
        super(DenseHidden, self).__init__()
        self.partial_model = Model(inputs=model.inputs, outputs=model.layers[hidden_layer].output)
        for layer in self.partial_model.layers:  # freeze model layers
            layer.trainable = False
        self.hidden_dim = hidden_dim
        if hidden_dim is not None:
            self.dense_layer = Dense(hidden_dim, activation=tf.nn.relu)
        self.output_layer = Dense(output_dim, activation=tf.nn.softmax)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.partial_model(x)
        x = Flatten()(x)
        if self.hidden_dim is not None:
            x = self.dense_layer(x)
        return self.output_layer(x)


class AdversarialAE(BaseDetector, FitMixin, ThresholdMixin):

    def __init__(self,
                 threshold: float = None,
                 ae: tf.keras.Model = None,
                 model: tf.keras.Model = None,
                 encoder_net: tf.keras.Sequential = None,
                 decoder_net: tf.keras.Sequential = None,
                 hidden_layer_kld: dict = None,
                 data_type: str = None
                 ) -> None:
        """
        Autoencoder (AE) based adversarial detector.

        Parameters
        ----------
        threshold
            Threshold used for adversarial score to determine adversarial instances.
        ae
            A trained tf.keras autoencoder model if available.
        model
            A trained tf.keras classification model.
        encoder_net
            Layers for the encoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        decoder_net
            Layers for the decoder wrapped in a tf.keras.Sequential class if no 'ae' is specified.
        hidden_layer_kld
            Dictionary with as keys the hidden layer(s) of the model which are extracted and used
            during training of the AE.
        data_type
            Optionally specifiy the data type (tabular, image or time-series). Added to metadata.
        """
        super().__init__()

        if threshold is None:
            logger.warning('No threshold level set. Need to infer threshold using `infer_threshold`.')

        self.threshold = threshold
        self.model = model
        for layer in self.model.layers:  # freeze model layers
            layer.trainable = False

        # check if model can be loaded, otherwise initialize AE model
        if isinstance(ae, tf.keras.Model):
            self.ae = ae
        elif isinstance(encoder_net, tf.keras.Sequential) and isinstance(decoder_net, tf.keras.Sequential):
            self.ae = AE(encoder_net, decoder_net)  # define AE model
        else:
            raise TypeError('No valid format detected for `ae` (tf.keras.Model) '
                            'or `encoder_net` and `decoder_net` (tf.keras.Sequential).')

        # intermediate feature map outputs for KLD
        if isinstance(hidden_layer_kld, dict):
            self.model_hl = []
            for hidden_layer, output_dim in hidden_layer_kld.items():
                self.model_hl.append(DenseHidden(self.model, hidden_layer, output_dim))
        else:
            self.model_hl = None

        # set metadata
        self.meta['detector_type'] = 'offline'
        self.meta['data_type'] = data_type

    def fit(self,
            X: np.ndarray,
            loss_fn: tf.keras.losses = loss_adv_vae,
            w_model: float = 1.,
            w_recon: float = 0.,
            w_hidden_model: list = None,
            optimizer: tf.keras.optimizers = tf.keras.optimizers.Adam(learning_rate=1e-3),
            epochs: int = 20,
            batch_size: int = 128,
            verbose: bool = True,
            log_metric: Tuple[str, "tf.keras.metrics"] = None,
            callbacks: tf.keras.callbacks = None,
            save_every: int = 1,
            save_path: str = None,
            preprocess: bool = False,
            temperature: float = 1.
            ) -> None:
        """
        Train Adversarial AE model.

        Parameters
        ----------
        X
            Training batch.
        loss_fn
            Loss function used for training.
        w_model
            Weight on model prediction loss term.
        w_recon
            Weight on elbo loss term.
        optimizer
            Optimizer used for training.
        epochs
            Number of training epochs.
        batch_size
            Batch size used for training.
        verbose
            Whether to print training progress.
        log_metric
            Additional metrics whose progress will be displayed if verbose equals True.
        callbacks
            Callbacks used during training.
        """
        # train arguments
        args = [self.ae, loss_fn, X]
        kwargs = {'optimizer': optimizer,
                  'epochs': epochs,
                  'batch_size': batch_size,
                  'verbose': verbose,
                  'log_metric': log_metric,
                  'callbacks': callbacks,
                  'save_every': save_every,
                  'save_path': save_path,
                  'preprocess': preprocess,
                  'loss_fn_kwargs': {'w_model': w_model,
                                     'w_recon': w_recon,
                                     'model': self.model,
                                     'hidden_model': self.model_hl,
                                     'w_hidden_model': w_hidden_model,
                                     'temperature': temperature,
                                     'loss_recon_type': 'mse'},
                  'hidden_model': self.model_hl
                  }

        # train
        trainer(*args, **kwargs)

    def infer_threshold(self,
                        X: np.ndarray,
                        threshold_perc: float = 95.
                        ) -> None:
        """
        Update threshold by a value inferred from the percentage of instances considered to be
        adversarial in a sample of the dataset.

        Parameters
        ----------
        X
            Batch of instances.
        threshold_perc
            Percentage of X considered to be normal based on the adversarial score.
        """
        # compute adversarial scores
        adv_score = self.score(X)

        # update threshold
        self.threshold = np.percentile(adv_score, threshold_perc)

    def score(self, X: np.ndarray, T: float = 1., scale_recon: bool = False,
              w_hidden_model: list = None) -> np.ndarray:
        """
        Compute adversarial scores.

        Parameters
        ----------
        X
            Batch of instances to analyze.
        T
            Temperature used for prediction probability scaling.

        Returns
        -------
        Array with adversarial scores for each instance in the batch.
        """
        # reconstructed instances
        X_recon = self.ae(X)

        # model predictions
        y = self.model(X)
        y_recon = self.model(X_recon)

        # scale predictions
        if T != 1.:
            y = y ** (1/T)
            y = y / tf.reshape(tf.reduce_sum(y, axis=-1), (-1, 1))

        if scale_recon:
            y_recon = y_recon ** (1/T)
            y_recon = y_recon / tf.reduce_sum(y_recon)

        adv_score = kld(y, y_recon).numpy()

        # hidden layer predictions
        if self.model_hl is not None:
            if w_hidden_model is None:
                w_hidden_model = list(np.ones(len(self.model_hl)))
            for hidden_m, w in zip(self.model_hl, w_hidden_model):
                h = hidden_m(X)
                h_recon = hidden_m(X_recon)
                adv_score += w * kld(h, h_recon).numpy()

        return adv_score

    def predict(self,
                X: np.ndarray,
                return_instance_score: bool = True) \
            -> Dict[Dict[str, str], Dict[np.ndarray, np.ndarray]]:
        """
        Predict whether instances are adversarial instances or not.

        Parameters
        ----------
        X
            Batch of instances.
        return_instance_score
            Whether to return instance level adversarial scores.

        Returns
        -------
        Dictionary containing 'meta' and 'data' dictionaries.
        'meta' has the model's metadata.
        'data' contains the adversarial predictions and instance level adversarial scores.
        """
        adv_score = self.score(X)

        # values above threshold are outliers
        adv_pred = (adv_score > self.threshold).astype(int)

        # populate output dict
        ad = adversarial_prediction_dict()
        ad['meta'] = self.meta
        ad['data']['is_adversarial'] = adv_pred
        if return_instance_score:
            ad['data']['instance_score'] = adv_score
        return ad

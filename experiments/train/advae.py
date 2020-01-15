import argparse
import tensorflow as tf
import yaml

from alibi_detect.ad import AdversarialVAE
from alibi_detect.utils.saving import load_tf_model
from experiments.configs.encoder_decoder import DATASETS

from official.vision.image_classification import resnet_cifar_model


def run(dataset: str) -> None:

    # load pretrained classifier
    clf_path = '../models/clf/' + dataset + '/'
    if dataset in ['mnist', 'fashion_mnist']:
        clf = load_tf_model(clf_path)
        preprocess = False
    elif dataset == 'cifar10':
        clf = tf.keras.models.load_model(clf_path + 'model.h5')
        preprocess = True

    # load and preprocess data
    if dataset == 'mnist':
        train, test = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        train, test = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        train, test = tf.keras.datasets.cifar10.load_data()
    X_train, y_train = train

    shape = (-1,) + X_train.shape[1:]
    if len(shape) == 3:
        shape += (1,)

    if dataset in ['mnist', 'fashion_mnist']:  # different preprocessing for CIFAR10
        X_train = X_train.reshape(shape).astype('float32') / 255

    # load experiment config file
    cfg_path = '../configs/vae_' + dataset + '.yml'
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # iterate over experiments in config file
    n_exp = len(list(cfg.keys()))
    for exp, params in cfg.items():

        print('Experiment {} of {}'.format(exp, n_exp))

        # initialize adversarial detector
        ad = AdversarialVAE(
            model=clf,
            encoder_net=DATASETS[dataset]['encoder_net'],
            decoder_net=DATASETS[dataset]['decoder_net'],
            latent_dim=params['latent_dim'],
            beta=params['beta']
        )

        # train VAE
        if not params['cov_elbo']:
            cov_elbo = None
        else:
            cov_elbo = {params['cov_elbo']['type']: params['cov_elbo']['var']}

        ad.fit(
            X_train,
            w_model=params['w_model'],
            w_recon=params['w_recon'],
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            cov_elbo=cov_elbo,
            epochs=params['epochs'],
            batch_size=128,
            verbose=params['verbose'],
            save_every=params['save_every'],
            save_path='../models/vae/' + dataset + '/' + str(exp) + '/',
            preprocess=preprocess
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Adversarial VAE.")
    parser.add_argument('--dataset', type=str, default='mnist')
    args = parser.parse_args()
    run(args.dataset)

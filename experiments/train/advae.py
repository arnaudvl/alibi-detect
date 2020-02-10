import argparse
import os
import numpy as np
import tensorflow as tf
import yaml

from alibi_detect.ad import AdversarialVAE, AdversarialAE
from alibi_detect.ad.adversarialae import HiddenKLD
from alibi_detect.utils.saving import load_tf_model
from experiments.configs.encoder_decoder import AE_ENC_DEC, VAE_ENC_DEC

from official.vision.image_classification import resnet_cifar_model


def run(dataset: str,
        detector: str,
        seeds: int = 1,
        start_exp: int = 0,
        end_exp: int = 1000,
        architecture: str = 'v0',
        weak_model: bool = False) -> None:

    # load pretrained classifier
    if weak_model:
        clf_path = '../models/clf/' + dataset + '_weak/'
    else:
        clf_path = '../models/clf/' + dataset + '/'
    if dataset in ['mnist', 'fashion_mnist']:
        clf = load_tf_model(clf_path)
        preprocess = False
    elif dataset == 'cifar10':
        if weak_model:
            clf = load_tf_model(clf_path)
        else:
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
    y_train = y_train.astype('int64').reshape(-1,)

    shape = (-1,) + X_train.shape[1:]
    if len(shape) == 3:
        shape += (1,)

    if dataset in ['mnist', 'fashion_mnist']:  # different preprocessing for CIFAR10
        X_train = X_train.reshape(shape).astype('float32') / 255

    # load experiment config file
    cfg_path = '../configs/' + dataset + '/' + detector + '_' + architecture + '.yml'
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # iterate over experiments in config file
    arch = dataset + '_' + architecture
    n_exp = len(list(cfg.keys()))
    for exp, params in cfg.items():

        if exp < start_exp:
            print('Experiment {} already ran'.format(exp))
            continue

        if exp > end_exp:
            print('Stop running experiments after experiment {}.'.format(exp))
            break

        print('Experiment {} of {}'.format(exp+1, n_exp))
        if detector == 'ae':
            params['beta'] = 'NA'
            params['loss_type'] = 'mse'
        try:
            print('Params: {} -- {} -- beta {} -- w_model {} -- w_recon {} -- Lrecon {} '
                  '-- Temp {} -- hl {} -- hl output dim {}'.format(
                dataset, params['ae_type'], params['beta'], params['w_model'],
                params['w_recon'], params['loss_type'], params['temperature'],
                params['hl'], params['hl_output_dim']))
        except KeyError:
            print('Params: {} -- {} -- beta {} -- w_model {} -- w_recon {} -- Lrecon {} '
                  '-- Temp {}'.format(
                dataset, params['ae_type'], params['beta'], params['w_model'],
                params['w_recon'], params['loss_type'], params['temperature']))

        for s in range(seeds):

            print('Seed {} of {}'.format(s+1, seeds))
            np.random.seed(s)
            tf.random.set_seed(s)

            if params['ae_type'] == 'vae':
                # initialize adversarial detector
                ad = AdversarialVAE(
                    model=clf,
                    encoder_net=VAE_ENC_DEC[arch]['encoder_net'],
                    decoder_net=VAE_ENC_DEC[arch]['decoder_net'],
                    latent_dim=params['latent_dim'],
                    beta=params['beta']
                )

                # train VAE
                if not params['cov_elbo']:
                    cov_elbo = None
                elif params['loss_type'] == 'elbo':
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
                    save_path='../models/'+detector+'/'+dataset+'/'+architecture+'/'+str(exp)+'_s'+str(s)+'/',
                    preprocess=preprocess,
                    loss_recon_type=params['loss_type']
                )

            elif params['ae_type'] == 'ae':
                # initialize hidden layer KLD models
                try:
                    model_hl = []
                    if not params['hidden_dim']:
                        params['hidden_dim'] = [None for _ in params['hl']]
                    for l, dim, hdim in zip(params['hl'], params['hl_output_dim'], params['hidden_dim']):
                        model_hl.append(HiddenKLD(clf, l, dim, hidden_dim=hdim))
                except KeyError:
                    model_hl = None

                # initialize adversarial detector
                ad = AdversarialAE(
                    model_hl=model_hl,
                    model=clf,
                    encoder_net=AE_ENC_DEC[arch]['encoder_net'],
                    decoder_net=AE_ENC_DEC[arch]['decoder_net'],
                )

                try:
                    params['w_hidden_model']
                except KeyError:
                    params['w_hidden_model'] = None

                if weak_model:
                    save_path = '../models/'+detector+'/'+dataset+'_weak/'+architecture+'/'+str(exp)+'_s'+str(s)+'/'
                else:
                    save_path = '../models/'+detector+'/'+dataset+'/'+architecture+'/'+str(exp)+'_s'+str(s)+'/'

                ad.fit(
                    X_train,
                    w_model=params['w_model'],
                    w_recon=params['w_recon'],
                    w_hidden_model=params['w_hidden_model'],
                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    epochs=params['epochs'],
                    batch_size=128,
                    verbose=params['verbose'],
                    save_every=params['save_every'],
                    save_path=save_path,
                    preprocess=preprocess,
                    temperature=params['temperature']
                )

                del ad.model_hl
                del ad


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Adversarial VAE.")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--detector', type=str, default='ae')
    parser.add_argument('--seeds', type=int, default=1)
    parser.add_argument('--start_exp', type=int, default=0)
    parser.add_argument('--end_exp', type=int, default=1000)
    parser.add_argument('--architecture', type=str, default='v0')
    parser.add_argument('--weak_model', type=bool, default=False)
    args = parser.parse_args()
    run(args.dataset, args.detector, args.seeds, args.start_exp, args.end_exp, args.architecture, args.weak_model)

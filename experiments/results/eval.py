import argparse
import numpy as np
import os
import pickle
import tensorflow as tf
from typing import Union
import yaml
from alibi_detect.ad import AdversarialAE, AdversarialVAE
from alibi_detect.ad.adversarialae import HiddenKLD
from alibi_detect.models.autoencoder import AE
from alibi_detect.utils.saving import load_tf_model, load_tf_vae
from official.vision.image_classification import resnet_cifar_model


def load_tf_ae(filepath: str) -> tf.keras.Model:
    model_dir = os.path.join(filepath, 'model')
    encoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'encoder_net.h5'))
    decoder_net = tf.keras.models.load_model(os.path.join(model_dir, 'decoder_net.h5'))
    ae = AE(encoder_net, decoder_net)
    try:
        ae.load_weights(os.path.join(model_dir, 'ae.ckpt'))
    except:
        ae.load_weights(os.path.join(model_dir, 'vae.ckpt'))
    return ae


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (y_true == y_pred).astype(int).sum() / y_true.shape[0]


def scale_by_instance(X: np.ndarray) -> np.ndarray:
    mean_ = X.mean(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    std_ = X.std(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    return (X - mean_) / std_


def predict_batch(model: tf.keras.Model, X: np.ndarray, batch_size: int = 256, recon: bool = False) -> np.ndarray:
    n = X.shape[0]
    if recon:
        shape = X.shape
        dtype = np.float32
    else:
        shape = (n,)
        dtype = np.int64
    preds = np.zeros(shape, dtype=dtype)
    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        if recon:
            preds[istart:istop] = model(X[istart:istop]).numpy()
        else:
            preds[istart:istop] = model(X[istart:istop]).numpy().argmax(axis=-1)
    return preds


def score_batch(detector: Union[AdversarialAE, AdversarialVAE],
                X: np.ndarray,
                batch_size: int = 256,
                temperature: float = 1.) -> np.ndarray:
    n = X.shape[0]
    scores = np.zeros((n,), dtype=np.float32)
    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        scores[istart:istop] = detector.score(X[istart:istop], T=temperature)
    return scores


def eval(dataset: str,
         attack: str,
         detector: str,
         architecture: str,
         start_exp: int = 0,
         end_exp: int = 1000,
         weak_model: bool = False) -> None:

    # load pretrained classifier
    if weak_model:
        clf_path = '../models/clf/' + dataset + '_weak/'
    else:
        clf_path = '../models/clf/' + dataset + '/'
    if dataset in ['mnist', 'fashion_mnist']:
        clf = load_tf_model(clf_path)
    elif dataset == 'cifar10':
        if weak_model:
            clf = load_tf_model(clf_path)
        else:
            clf = tf.keras.models.load_model(clf_path + 'model.h5')

    # load original dataset
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    shape = (-1,) + X_train.shape[1:]
    if len(shape) == 3:  # add channels dim
        shape += (1,)
    X_train = X_train.reshape(shape).astype('float32')
    X_test = X_test.reshape(shape).astype('float32')
    y_train = y_train.astype('int64').reshape(-1,)
    y_test = y_test.astype('int64').reshape(-1,)

    # preprocess data
    if dataset in ['mnist', 'fashion_mnist']:
        X_train /= 255
        X_test /= 255
    elif dataset == 'cifar10':
        X_train = scale_by_instance(X_train)
        X_test = scale_by_instance(X_test)

    # load adversarial dataset
    if weak_model:
        data_path = '../datasets/' + dataset + '_weak/' + attack + '.npz'
    else:
        data_path = '../datasets/' + dataset + '/' + attack + '.npz'
    adv_data = np.load(data_path)
    adv_train, adv_test = adv_data['X_train_adv'], adv_data['X_test_adv']
    y_train, y_test = adv_data['y_train'], adv_data['y_test']

    # classifier accuracy on original instances
    preds_orig_train = predict_batch(clf, X_train)
    preds_orig_test = predict_batch(clf, X_test)
    acc_orig_train = accuracy(y_train, preds_orig_train)
    acc_orig_test = accuracy(y_test, preds_orig_test)

    # classifier accuracy on adversarial instances -- no defense
    preds_nodefense_train = predict_batch(clf, adv_train)
    preds_nodefense_test = predict_batch(clf, adv_test)
    acc_nodefense_train = accuracy(y_train, preds_nodefense_train)
    acc_nodefense_test = accuracy(y_test, preds_nodefense_test)

    # create list of directories with detectors
    if weak_model:
        detector_path = '../models/' + detector + '/' + dataset + '_weak/' + architecture + '/'
    else:
        detector_path = '../models/' + detector + '/' + dataset + '/' + architecture + '/'
    exp_paths = []
    for path, _, _ in os.walk(detector_path):
        if (path.endswith('model') or path.endswith('s0') or path.endswith('s1')
                or path.endswith('v0/') or path.endswith('v1/') or path.endswith('v2/')
                or path.endswith('s2') or path.endswith('v3/') or path.endswith('v4/')
                or path.endswith('js') or path.endswith('v5/') or path.endswith('v6/')):
            continue
        else:
            exp_paths.append(path)
    print(exp_paths)

    # load experiment config file
    cfg_path = '../configs/' + dataset + '/' + detector + '_' + architecture + '.yml'
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    print(cfg)

    # compute performance metrics for all experiments
    acc = {}
    preds = {}
    scores = {}
    acc_best = {'orig': {'train': acc_orig_train, 'test': acc_orig_test},
                'nodefense': {'train': acc_nodefense_train, 'test': acc_nodefense_test},
                'defense': {'train': -1, 'test': -1},
                'best': '-1'}
    acc_worst = {'orig': {'train': acc_orig_train, 'test': acc_orig_test},
                 'nodefense': {'train': acc_nodefense_train, 'test': acc_nodefense_test},
                 'defense': {'train': 1., 'test': 1.},
                 'worst': '-1'}
    if detector == 'vae':
        state_dict = {'latent_dim': cfg[0]['latent_dim'], 'beta': 0}  # beta is not relevant for inference
    n_exp = len(exp_paths)
    for i, exp in enumerate(exp_paths):

        print(exp)

        try:
            istart = exp[exp.find('_s') - 2]
            iend = exp[exp.find('_s') - 1]
            icfg = int(istart + iend)
        except:
            icfg = int(exp[exp.find('_s') - 1])

        if icfg < start_exp:
            print('Start running results from experiment {}'.format(start_exp))
            continue

        if icfg > end_exp:
            print('Stop running results after experiment {}'.format(end_exp))
            continue

        print('Running config {}...'.format(icfg))

        # load adversarial (V)AE and initialize detector
        if detector == 'vae':
            det = load_tf_vae(exp, state_dict)
            ad = AdversarialVAE(threshold=None, model=clf, vae=det, samples=5)
        elif detector == 'ae':
            params = cfg[icfg]
            try:
                model_hl = []
                model_dir = os.path.join(exp, 'model')
                for j, (l, dim, hdim) in enumerate(zip(params['hl'], params['hl_output_dim'], params['hidden_dim'])):
                    m_hl = HiddenKLD(clf, l, dim, hidden_dim=hdim)
                    m_hl.load_weights(os.path.join(model_dir, 'model_hl_' + str(j) + '.ckpt'))
                    model_hl.append(m_hl)
                print('KL-divergence hidden layers loaded.')
            except KeyError:
                model_hl = None
                print('No KL-divergence on hidden layers.')
            det = load_tf_ae(exp)
            ad = AdversarialAE(
                model_hl=model_hl,
                threshold=None,
                model=clf,
                ae=det
            )

        # classifier accuracy on reconstructed instances -- adversarial defense
        adv_recon_train = predict_batch(det, adv_train, recon=True)
        adv_recon_test = predict_batch(det, adv_test, recon=True)
        preds_defense_train = predict_batch(clf, adv_recon_train)
        preds_defense_test = predict_batch(clf, adv_recon_test)
        acc_defense_train = accuracy(y_train, preds_defense_train)
        acc_defense_test = accuracy(y_test, preds_defense_test)

        # store accuracies
        acc_exp = {'orig': {'train': acc_orig_train, 'test': acc_orig_test},
                   'nodefense': {'train': acc_nodefense_train, 'test': acc_nodefense_test},
                   'defense': {'train': acc_defense_train, 'test': acc_defense_test}}

        if acc_defense_test > acc_best['defense']['test']:
            acc_best['defense']['test'] = acc_defense_test
            acc_best['defense']['train'] = acc_defense_train
            acc_best['best'] = exp

        if acc_defense_test < acc_worst['defense']['test']:
            acc_worst['defense']['test'] = acc_defense_test
            acc_worst['defense']['train'] = acc_defense_train
            acc_worst['worst'] = exp

        # store predictions
        preds_exp = {'orig': {'train': preds_orig_train, 'test': preds_orig_test},
                     'nodefense': {'train': preds_nodefense_train, 'test': preds_nodefense_test},
                     'defense': {'train': preds_defense_train, 'test': preds_defense_test}}

        # 2. adversarial scores
        # original instances
        try:
            T = cfg[icfg]['temperature']
        except KeyError:
            T = 1.
        score_orig_train = score_batch(ad, X_train, temperature=T)
        score_orig_test = score_batch(ad, X_test, temperature=T)

        # adversarial instances
        score_adv_train = score_batch(ad, adv_train, temperature=T)
        score_adv_test = score_batch(ad, adv_test, temperature=T)

        # store scores
        scores_exp = {'orig': {'train': score_orig_train, 'test': score_orig_test},
                      'adv': {'train': score_adv_train, 'test': score_adv_test}}

        # store metrics
        acc[exp] = acc_exp
        preds[exp] = preds_exp
        scores[exp] = scores_exp

        print('Conf {} of {} -- {} -- test acc: orig {:.4f} -- no defense {:.4f} -- defense {:.4f} '
              '-- best {:.4f} -- worst {:.4f}'.format(
            i, n_exp, exp, acc_orig_test, acc_nodefense_test, acc_defense_test,
            acc_best['defense']['test'], acc_worst['defense']['test'])
        )

    # save metrics
    if weak_model:
        save_path = './' + detector + '/' + dataset + '_weak/' + attack + '/' + architecture + '.pickle'
    else:
        save_path = './' + detector + '/' + dataset + '/' + attack + '/' + architecture + '_wrecon1.pickle'
    #save_path = './' + detector + '/' + dataset + '/' + attack + '/' + architecture + '_wrecon1.pickle'
    with open(save_path, 'wb') as f:
        pickle.dump([acc, preds, scores, acc_best, acc_worst], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute metrics.")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--attack', type=str, default='fgsm')
    parser.add_argument('--detector', type=str, default='vae')
    parser.add_argument('--architecture', type=str, default='v0')
    parser.add_argument('--start_exp', type=int, default=0)
    parser.add_argument('--end_exp', type=int, default=1000)
    parser.add_argument('--weak_model', type=bool, default=False)
    args = parser.parse_args()
    eval(args.dataset, args.attack, args.detector, args.architecture, args.start_exp, args.end_exp,
         args.weak_model)

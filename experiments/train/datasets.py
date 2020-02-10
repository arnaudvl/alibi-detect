import argparse
import foolbox
from foolbox.models import TensorFlowEagerModel
from foolbox.attacks import FGSM, CarliniWagnerL2Attack, SparseL1BasicIterativeAttack
import numpy as np
import pickle
import tensorflow as tf
from typing import Callable, Union
from alibi_detect.ad.adversarialae import DefenseWhiteBox
from alibi_detect.utils.saving import load_tf_model
from official.vision.image_classification import resnet_cifar_model


def scale_by_instance(X: np.ndarray) -> np.ndarray:
    mean_ = X.mean(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    std_ = X.std(axis=(1, 2, 3)).reshape(-1, 1, 1, 1)
    return (X - mean_) / std_, mean_, std_


def adversarial_dataset(clf: tf.keras.Model,
                        X: np.ndarray,
                        y: np.ndarray,
                        attack: Callable,
                        batch_size: int = 256,
                        bounds: tuple = (0, 1),
                        preprocessing: Union[dict, tuple] = (0, 1),
                        num_classes: int = 10,
                        kwargs: dict = None
                        ) -> np.ndarray:
    # initialize attack
    model = TensorFlowEagerModel(clf,
                                 bounds=bounds,
                                 preprocessing=preprocessing,
                                 num_classes=num_classes)
    att = attack(model)

    # apply attack per batch
    adv = np.zeros_like(X)
    n = X.shape[0]
    n_minibatch = int(np.ceil(n / batch_size))
    for i in range(n_minibatch):
        istart, istop = i * batch_size, min((i + 1) * batch_size, n)
        try:
            if kwargs is None:
                adv[istart:istop] = att(X[istart:istop], y[istart:istop])
            else:
                adv[istart:istop] = att(X[istart:istop], y[istart:istop], **kwargs)
        except AssertionError:  # catch weird sl1bia error for cifar10
            adv[istart:istop] = X[istart:istop]
    return adv


def run(dataset: str,
        attack_type: str = 'cw',
        white_box: bool = False,
        attack_train: bool = True,
        weak_model: bool = False,
        max_eps: float = 0.3) -> None:

    # load pretrained classifier
    if weak_model:
        clf_path = '../models/clf/' + dataset + '_weak/'
    else:
        clf_path = '../models/clf/' + dataset + '/'
    if dataset in ['mnist', 'fashion_mnist']:
        if white_box:
            print('Loading white box attack model...')
            if dataset == 'mnist':
                ae_path = '../models/ae/mnist/v2/0_s1/36'
            elif dataset == 'fashion_mnist':
                ae_path = '../models/ae/fashion_mnist/v2/0_s1/48'
            clf = DefenseWhiteBox(clf=clf_path, ae=ae_path)
            print('...loaded!')
        else:
            clf = load_tf_model(clf_path)
    elif dataset == 'cifar10':
        if weak_model:
            ae_path = '../models/ae/cifar10_weak/v1/0_s0/48'
            clf = load_tf_model(clf_path)
        else:
            ae_path = '../models/ae/cifar10/v1/0_s0/78'
            clf = tf.keras.models.load_model(clf_path + 'model.h5')
        if white_box:
            print('Loading white box attack model...')
            clf = DefenseWhiteBox(clf=clf, ae=ae_path)
            print('...loaded!')

    # load and preprocess data
    kwargs = None
    if dataset == 'mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        if attack_type == 'cw':
            kwargs = {
                'binary_search_steps': 7,
                'max_iterations': 200,
                'learning_rate': 1e-1,
                'initial_const': 100
            }
        elif attack_type == 'sl1bia':
            kwargs = {
                'q': 80,  # percentile
                'binary_search': True,
                'epsilon': 0.1,  # l1-bound
                'stepsize': 0.05,  # gamma
                'iterations': 10,
                'random_start': False,
                'return_early': True
            }
    elif dataset == 'fashion_mnist':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        if attack_type == 'cw':
            kwargs = {
                'binary_search_steps': 9,
                'max_iterations': 200,
                'learning_rate': 1e-1,
                'initial_const': 100
            }
        elif attack_type == 'sl1bia':
            kwargs = {
                'q': 80,  # percentile
                'binary_search': True,
                'epsilon': 0.1,  # l1-bound
                'stepsize': 0.05,  # gamma
                'iterations': 10,
                'random_start': False,
                'return_early': True
            }
    elif dataset == 'cifar10':
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        if attack_type == 'cw':
            kwargs = {
                'binary_search_steps': 9,
                'max_iterations': 100,
                'learning_rate': 1e-2,
                'initial_const': 1
            }
        elif attack_type == 'sl1bia':
            kwargs = {
                'q': 80,  # percentile
                'binary_search': True,
                'epsilon': 0.1,  # l1-bound
                'stepsize': 0.05,  # gamma
                'iterations': 10,
                'random_start': False,
                'return_early': True
            }

    y_train = y_train.astype(np.int64).reshape(-1,)
    y_test = y_test.astype(np.int64).reshape(-1,)

    shape = (-1,) + X_train.shape[1:]
    if len(shape) == 3:
        shape += (1,)

    X_train = X_train.reshape(shape).astype('float32')
    X_test = X_test.reshape(shape).astype('float32')

    if dataset in ['mnist', 'fashion_mnist']:  # different preprocessing for CIFAR10
        X_train /= 255
        X_test /= 255
        bounds = (0, 1)
        preprocessing = (0, 1)
    elif dataset == 'cifar10':
        X_train = scale_by_instance(X_train)[0]
        X_test = scale_by_instance(X_test)[0]
        bounds = (min(X_train.min(), X_test.min()), max(X_train.max(), X_test.max()))
        preprocessing = dict(mean=[0., 0., 0.], std=[1., 1., 1.], axis=-1)

    # attack
    if attack_type == 'fgsm':
        attack = FGSM
    elif attack_type == 'cw':
        attack = CarliniWagnerL2Attack
    elif attack_type == 'sl1bia':
        attack = SparseL1BasicIterativeAttack

    if attack_type == 'fgsm':
        kwargs = {
            'epsilons': 1000,
            'max_epsilon': max_eps
        }

    if attack_train:
        print('Attacking train set...')
        X_train_adv = adversarial_dataset(clf,
                                          X_train,
                                          y_train,
                                          attack,
                                          batch_size=256,
                                          bounds=bounds,
                                          preprocessing=preprocessing,
                                          kwargs=kwargs)
        print('Done!')

    print('Attacking test set...')
    X_test_adv = adversarial_dataset(clf,
                                     X_test,
                                     y_test,
                                     attack,
                                     batch_size=256,
                                     bounds=bounds,
                                     preprocessing=preprocessing,
                                     kwargs=kwargs)
    print('Done!')

    # identify unsuccessful attacks and replace NaN's with original data
    if attack_train:
        idx_nan_train = np.unique(np.where(np.isnan(X_train_adv))[0])
        X_train_adv[idx_nan_train] = X_train[idx_nan_train]
        acc_train = len(idx_nan_train) / X_train.shape[0]
        print('Accuracy train after attack: {:.4f}'.format(acc_train))

    idx_nan_test = np.unique(np.where(np.isnan(X_test_adv))[0])
    X_test_adv[idx_nan_test] = X_test[idx_nan_test]
    acc_test = len(idx_nan_test) / X_test.shape[0]
    print('Accuracy test after attack: {:.4f}'.format(acc_test))

    # save data
    if attack_type == 'fgsm':
        if weak_model:
            save_path = '../datasets/' + dataset + '_weak/' + attack_type + '_' + str(max_eps)[-1]
        else:
            save_path = '../datasets/' + dataset + '/' + attack_type + '_' + str(max_eps)[-1]
    else:
        if weak_model:
            save_path = '../datasets/' + dataset + '_weak/' + attack_type
        else:
            save_path = '../datasets/' + dataset + '/' + attack_type

    if white_box:
        save_path = save_path + '_whitebox'

    if attack_train:
        np.savez_compressed(save_path,
                            no_adv_train=idx_nan_train,
                            no_adv_test=idx_nan_test,
                            X_train_adv=X_train_adv,
                            X_test_adv=X_test_adv,
                            y_train=y_train,
                            y_test=y_test)
    else:
        np.savez_compressed(save_path,
                            no_adv_test=idx_nan_test,
                            X_test_adv=X_test_adv,
                            y_test=y_test)

    if kwargs is None:
        kwargs = {'params': 'default'}

    kwargs['attack_type'] = attack_type
    with open(save_path + '_meta.pickle', 'wb') as f:
        pickle.dump(kwargs, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create adversarial datasets.")
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--attack_type', type=str, default='fgsm')
    parser.add_argument('--white_box', type=bool, default=True)
    parser.add_argument('--attack_train', type=bool, default=True)
    parser.add_argument('--weak_model', type=bool, default=False)
    parser.add_argument('--max_eps', type=float, default=0.3)
    args = parser.parse_args()
    run(args.dataset, args.attack_type, args.white_box, args.attack_train, args.weak_model, args.max_eps)

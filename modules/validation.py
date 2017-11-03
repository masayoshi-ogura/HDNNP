# -*- coding: utf-8 -*-

from config import hp

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


def kfold(nsample):
    if hp.split.__class__ is not int or not hp.split > 0:
        raise 'When the validation method is K-Fold, `hp.split` must be positive integer.'

    indices = np.array(range(nsample))
    kf = KFold(n_splits=hp.split, shuffle=True, random_state=None)
    for training_indices, validation_indices in kf.split(indices):
        yield training_indices, validation_indices


def holdout(nsample):
    if hp.split.__class__ is not list or len(hp.split) != 2:
        raise 'When the validation method is Hold-Out, `hp.split` must be python list. [train_size, test_size]'

    indices = np.array(range(nsample))
    train_size = float(hp.split[0]) / sum(hp.split)
    test_size = float(hp.split[1]) / sum(hp.split)
    training_indices, validation_indices = train_test_split(indices, train_size=train_size, test_size=test_size)
    yield training_indices, validation_indices


VALIDATIONS = {'kfold': kfold, 'holdout': holdout}

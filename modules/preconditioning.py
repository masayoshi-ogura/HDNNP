# -*- coding: utf-8 -*-

import numpy as np
from sklearn import decomposition


class PreconditionBase(object):
    def __init__(self):
        pass

    def decompose(self, dataset):
        pass


class PCA(PreconditionBase):
    def __init__(self, threshold=0.999):
        self._threshold = threshold

    def decompose(self, dataset):
        input = dataset.input
        dinput = dataset.dinput
        if not hasattr(self, '_components'):
            pca = decomposition.PCA()
            pca.fit(input.reshape(-1, input.shape[-1]))
            filter = np.add.accumulate(pca.explained_variance_ratio_) < self._threshold
            self._mean = input.mean(axis=(0, 1))
            self._components = pca.components_[filter].T
            self._ncomponent = self._components.shape[1]
            print 'decompose from {} to {}'.format(self._components.shape[0], self._components.shape[1])

        dataset.input = np.dot(input - self._mean, self._components)
        dataset.dinput = np.tensordot(dinput - self._mean, self._components, ((3,), (0,)))
        dataset.ninput = self._ncomponent


PRECOND = {None: PreconditionBase, 'pca': PCA}

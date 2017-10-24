# -*- coding: utf-8 -*-

import numpy as np
from sklearn import decomposition


class PreconditionBase(object):
    def __init__(self):
        pass

    def decompose(self, dataset):
        pass


class PCA(PreconditionBase):
    def __init__(self, threshold=0.9999):
        self._threshold = threshold

    def decompose(self, dataset):
        if not hasattr(self, '_components'):
            self._mean = {}
            self._components = {}
            self._ncomponent = 0
            for symbol, indices in dataset.composition['index'].items():
                X = dataset.input[:, list(indices), :].reshape(-1, dataset.input.shape[-1])
                pca = decomposition.PCA()
                pca.fit(X)
                self._mean[symbol] = X.mean(axis=0)
                self._components[symbol] = pca.components_
                self._ncomponent = max(self._ncomponent,
                                       sum(np.add.accumulate(pca.explained_variance_ratio_) < self._threshold))
            # adjust ncomponent to max of it
            print 'decompose from {} to {}'.format(dataset.input.shape[-1], self._ncomponent)
            for symbol, component in self._components.items():
                self._components[symbol] = component[:self._ncomponent].T

        mean = np.array([self._mean[symbol]
                         for symbol in dataset.composition['symbol']])
        components = np.array([self._components[symbol]
                               for symbol in dataset.composition['symbol']])
        dataset.input = np.einsum('ijk,jkl->ijl', dataset.input - mean, components)
        dataset.dinput = np.einsum('ijkl,jlm->ijkm', dataset.dinput - mean[:, None, :], components)
        dataset.ninput = self._ncomponent


PRECOND = {None: PreconditionBase, 'pca': PCA}

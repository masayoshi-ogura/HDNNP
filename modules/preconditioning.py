# -*- coding: utf-8 -*-

import numpy as np
from sklearn import decomposition

from util import mpiprint


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
            for element, indices in dataset.composition.index.items():
                X = dataset.input[:, list(indices), :].reshape(-1, dataset.input.shape[-1])
                pca = decomposition.PCA()
                pca.fit(X)
                self._mean[element] = X.mean(axis=0)
                self._components[element] = pca.components_
                self._ncomponent = max(self._ncomponent,
                                       sum(np.add.accumulate(pca.explained_variance_ratio_) < self._threshold))
            # adjust ncomponent to max of it
            for element, component in self._components.items():
                self._components[element] = component[:self._ncomponent].T

        mean = np.array([self._mean[element]
                         for element in dataset.composition.element])
        components = np.array([self._components[element]
                               for element in dataset.composition.element])
        new_input = np.einsum('ijk,jkl->ijl', dataset.input - mean, components)
        new_dinput = np.einsum('ijkl,jlm->ijkm', dataset.dinput - mean[:, None, :], components)
        dataset.reset_inputs(new_input, new_dinput)


PRECOND = {None: PreconditionBase, 'pca': PCA}

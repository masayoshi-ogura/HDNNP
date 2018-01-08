# -*- coding: utf-8 -*-

from os import path
import numpy as np
from sklearn import decomposition


class PreconditionBase(object):
    def __init__(self, *args):
        pass

    def load(self, *args):
        pass

    def save(self, *args):
        pass

    def decompose(self, *args):
        pass


class PCA(PreconditionBase):
    def __init__(self, ncomponent):
        self._ncomponent = ncomponent
        self._mean = {}
        self._components = {}
        self._elements = []

    def load(self, filename):
        if path.exists(filename):
            with np.load(filename) as ndarray:
                elements = ndarray['elements']
                self._mean = {element: ndarray['mean/{}'.format(element)] for element in elements}
                self._components = {element: ndarray['components/{}'.format(element)] for element in elements}

    def save(self, filename):
        if not path.exists(filename):
            mean_dict = {'mean/{}'.format(k): v for k, v in self._mean.items()}
            components_dict = {'components/{}'.format(k): v for k, v in self._components.items()}
            dic = dict(mean_dict.items() + components_dict.items() + [('elements', self._elements)])
            np.savez(filename, **dic)

    def decompose(self, dataset):
        # # !!! USING THRESHOLD !!!
        # elements = dataset.composition.index.keys()
        # if elements != self._elements:
        #     ncomponent = 0
        #     for element, indices in dataset.composition.index.items():
        #         X = dataset.input[:, list(indices), :].reshape(-1, dataset.input.shape[-1])
        #         pca = decomposition.PCA()
        #         pca.fit(X)
        #         self._mean[element] = X.mean(axis=0)
        #         self._components[element] = pca.components_.astype(np.float32)
        #         ncomponent = max(ncomponent,
        #                          sum(np.add.accumulate(pca.explained_variance_ratio_) < self._threshold))
        #     # adjust ncomponent to max of it
        #     for element, component in self._components.items():
        #         self._components[element] = component[:ncomponent].T
        for element, indices in dataset.composition.index.items():
            if element in self._elements:
                continue

            X = dataset.input[:, list(indices), :].reshape(-1, dataset.input.shape[-1])
            pca = decomposition.PCA()
            pca.fit(X)
            self._mean[element] = X.mean(axis=0)
            self._components[element] = pca.components_[:self._ncomponent].T.astype(np.float32)
            self._elements.append(element)

        mean = np.array([self._mean[element]
                         for element in dataset.composition.element])
        components = np.array([self._components[element]
                               for element in dataset.composition.element])
        new_input = np.einsum('ijk,jkl->ijl', dataset.input - mean, components)
        new_dinput = np.einsum('ijkl,jlm->ijkm', dataset.dinput - mean[:, None, :], components)
        dataset.reset_inputs(new_input, new_dinput)


PRECOND = {None: PreconditionBase, 'pca': PCA}

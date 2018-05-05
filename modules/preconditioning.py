# -*- coding: utf-8 -*-

from os import path
import numpy as np
from sklearn import decomposition


class PreconditionBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def decompose(self, *args, **kwargs):
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
                self._elements += list(elements)

    def save(self, filename):
        if not path.exists(filename):
            mean_dict = {'mean/{}'.format(k): v for k, v in self._mean.iteritems()}
            components_dict = {'components/{}'.format(k): v for k, v in self._components.iteritems()}
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
        for element, indices in dataset.composition.index.iteritems():
            if element in self._elements:
                continue

            X = dataset.input[:, list(indices), :].reshape(-1, dataset.input.shape[-1])
            pca = decomposition.PCA(n_components=self._ncomponent)
            pca.fit(X)
            self._mean[element] = pca.mean_.astype(np.float32)
            self._components[element] = pca.components_.T.astype(np.float32)
            self._elements.append(element)

        mean = np.array([self._mean[element]  # (atom, feature)
                         for element in dataset.composition.element])
        components = np.array([self._components[element]  # (atom, feature, component)
                               for element in dataset.composition.element])
        new_input = np.einsum('ijk,jkl->ijl', dataset.input - mean[None, :, :], components)  # (sample, atom, feature)
        new_dinput = np.einsum('ijkmn,jkl->ijlmn', dataset.dinput - mean[None, :, :, None, None], components)  # (sample, atom, feature, atom, 3)
        dataset.reset_inputs(new_input, new_dinput)


PRECOND = {'none': PreconditionBase, 'pca': PCA}

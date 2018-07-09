# -*- coding: utf-8 -*-

from os import path
import numpy as np
from sklearn import decomposition

from .util import pprint


class PreprocBase(object):
    def __init__(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def decompose(self, *args, **kwargs):
        pass


class PCA(PreprocBase):
    def __init__(self, ncomponent=20, *args, **kwargs):
        super(PCA, self).__init__(*args, **kwargs)
        self._ncomponent = ncomponent
        self._mean = {}
        self._components = {}
        self._elements = []

    @property
    def mean(self):
        return self._mean

    @property
    def components(self):
        return self._components

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
        for element in dataset.composition.element:
            if element in self._elements:
                continue

            nfeature = dataset.input.shape[-1]
            X = dataset.input.take(dataset.composition.indices[element], 1).reshape(-1, nfeature)
            pca = decomposition.PCA(n_components=self._ncomponent)
            pca.fit(X)
            self._mean[element] = pca.mean_.astype(np.float32)
            self._components[element] = pca.components_.T.astype(np.float32)
            self._elements.append(element)
            pprint('Initialize PCA parameters of element: {}\n'
                   '\tdecompose SF from {} to {}.\n\tcumulative contribution rate: {}'
                   .format(element, nfeature, self._ncomponent, np.sum(pca.explained_variance_ratio_)))

        mean = np.array([self._mean[element]  # (atom, feature)
                         for element in dataset.composition.atom])
        components = np.array([self._components[element]  # (atom, feature, component)
                               for element in dataset.composition.atom])
        dataset.input = np.einsum('ijk,jkl->ijl', dataset.input - mean, components)  # (sample, atom, feature)
        dataset.dinput = np.einsum('ijkmn,jkl->ijlmn', dataset.dinput, components)  # (sample, atom, feature, atom, 3)


PREPROC = {None: PreprocBase, 'pca': PCA}

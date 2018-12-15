# -*- coding: utf-8 -*-

__all__ = [
    'PCA',
    ]

import numpy as np
from sklearn import decomposition

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class PCA(PreprocessBase):
    def __init__(self, n_components):
        super().__init__()
        self._elements = set()
        self._n_components = n_components
        self._mean = {}
        self._transform = {}

    @property
    def elements(self):
        return sorted(self._elements)

    @property
    def n_components(self):
        return self._n_components

    @property
    def mean(self):
        return self._mean

    @property
    def transform(self):
        return self._transform

    def apply(self, dataset, elemental_composition, verbose=True):
        assert len(dataset) < 3

        self._initialize_params(dataset[0], elemental_composition, verbose)

        mean = np.array(
            [self._mean[element] for element in elemental_composition])
        transform = np.array(
            [self._transform[element] for element in elemental_composition])

        if len(dataset) >= 0:
            dataset[0] = np.einsum('ijk,jkl->ijl', dataset[0]-mean, transform)
        if len(dataset) >= 1:
            dataset[1] = np.einsum('ijkmn,jkl->ijlmn', dataset[1], transform)

        return dataset

    def dump_params(self):
        # todo
        return

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            self._elements = ndarray['elements'].item()
            self._n_components = ndarray['n_components'].item()
            self._mean = {element: ndarray[f'mean:{element}']
                          for element in self._elements}
            self._transform = {element: ndarray[f'transform:{element}']
                               for element in self._elements}
            if verbose:
                pprint(f'Loaded PCA parameters from {file_path}.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            info = {
                'elements': self._elements,
                'n_components': self._n_components,
                }
            mean = {f'mean:{k}': v for k, v in self._mean.items()}
            transform = {f'transform:{k}': v
                         for k, v in self._transform.items()}
            np.savez(file_path, **info, **mean, **transform)
            if verbose:
                pprint(f'Saved PCA parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        for element in set(elemental_composition) - self._elements:
            nfeature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, nfeature)
            pca = decomposition.PCA(n_components=self._n_components)
            pca.fit(X)
            self._elements.add(element)
            self._mean[element] = pca.mean_.astype(np.float32)
            self._transform[element] = pca.components_.T.astype(np.float32)
            if verbose:
                pprint(f'''
Initialized PCA parameters for {element}
    Feature dimension: {nfeature} => {self._n_components}
    Cumulative contribution rate = {np.sum(pca.explained_variance_ratio_)}
''')

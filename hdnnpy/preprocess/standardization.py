# coding: utf-8

import numpy as np

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class Standardization(PreprocessBase):
    name = 'standardization'

    def __init__(self):
        super().__init__()
        self._mean = {}
        self._std = {}

    @property
    def mean(self):
        return self._mean

    @property
    def std(self):
        return self._std

    def apply(self, dataset, elemental_composition, verbose=True):
        order = len(dataset) - 1
        assert 0 <= order <= 1

        self._initialize_params(dataset[0], elemental_composition, verbose)

        mean = np.array(
            [self._mean[element] for element in elemental_composition])
        std = np.array(
            [self._std[element] for element in elemental_composition])

        if order >= 0:
            dataset[0] -= mean
            dataset[0] /= std
        if order >= 1:
            dataset[1] /= std[..., None, None]

        return dataset

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            self._elements = ndarray['elements'].item()
            self._mean = {element: ndarray[f'mean:{element}']
                          for element in self._elements}
            self._std = {element: ndarray[f'std:{element}']
                         for element in self._elements}
            if verbose:
                pprint(f'Loaded Standardization parameters from {file_path}.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            info = {'elements': self._elements}
            mean = {f'mean:{k}': v for k, v in self._mean.items()}
            std = {f'std:{k}': v for k, v in self._std.items()}
            np.savez(file_path, **info, **mean, **std)
            if verbose:
                pprint(f'Saved Standardization parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        for element in set(elemental_composition) - self._elements:
            n_feature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, n_feature)
            self._elements.add(element)
            self._mean[element] = X.mean(axis=0, dtype=np.float32)
            self._std[element] = X.std(axis=0, ddof=1, dtype=np.float32)
            if verbose:
                pprint(f'Initialized Standardization parameters for {element}')

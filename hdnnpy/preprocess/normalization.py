# coding: utf-8

import numpy as np

from hdnnpy.preprocess.preprocess_base import PreprocessBase
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class Normalization(PreprocessBase):
    name = 'normalization'

    def __init__(self):
        super().__init__()
        self._elements = set()
        self._max = {}
        self._min = {}

    @property
    def elements(self):
        return sorted(self._elements)

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    def apply(self, dataset, elemental_composition, verbose=True):
        order = len(dataset) - 1
        assert 0 <= order <= 1

        self._initialize_params(dataset[0], elemental_composition, verbose)

        max_ = np.array(
            [self._max[element] for element in elemental_composition])
        min_ = np.array(
            [self._min[element] for element in elemental_composition])

        if order >= 0:
            dataset[0] -= min_
            dataset[0] /= max_ - min_
        if order >= 1:
            dataset[1] /= (max_ - min_)[..., None, None]

        return dataset

    def dump_params(self):
        # todo
        return

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            self._elements = ndarray['elements'].item()
            self._max = {element: ndarray[f'max:{element}']
                         for element in self._elements}
            self._min = {element: ndarray[f'min:{element}']
                         for element in self._elements}
            if verbose:
                pprint(f'Loaded Normalization parameters from {file_path}.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            info = {'elements': self._elements}
            max_ = {f'max:{k}': v for k, v in self._max.items()}
            min_ = {f'min:{k}': v for k, v in self._min.items()}
            np.savez(file_path, **info, **max_, **min_)
            if verbose:
                pprint(f'Saved Normalization parameters to {file_path}.')

    def _initialize_params(self, data, elemental_composition, verbose):
        for element in set(elemental_composition) - self._elements:
            n_feature = data.shape[2]
            mask = np.array(elemental_composition) == element
            X = data[:, mask].reshape(-1, n_feature)
            self._elements.add(element)
            self._max[element] = X.max(axis=0, dtype=np.float32)
            self._min[element] = X.min(axis=0, dtype=np.float32)
            if verbose:
                pprint(f'Initialized Normalization parameters for {element}')

# coding: utf-8

from abc import (ABC, abstractmethod)

import numpy as np

from hdnnpy.utils import (MPI, pprint)


class DescriptorDatasetBase(ABC):
    DESCRIPTORS = []
    name = ''

    def __init__(self, order, structures):
        self._order = order
        self._descriptors = self.DESCRIPTORS[: order+1]
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._structures = structures
        self._tag = structures[0].info['tag']
        self._dataset = []
        self._feature_keys = []

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                index = self._descriptors.index(item)
            except ValueError:
                raise KeyError(item) from None
            return self._dataset[index]
        else:
            return [data[item] for data in self._dataset]

    def __len__(self):
        return len(self._structures)

    @property
    def descriptors(self):
        return self._descriptors

    @property
    def elemental_composition(self):
        return self._elemental_composition

    @property
    def elements(self):
        return self._elements

    @property
    def feature_keys(self):
        return self._feature_keys

    @property
    def has_data(self):
        return len(self._dataset) != 0

    @property
    def n_feature(self):
        return len(self._feature_keys)

    @property
    def order(self):
        return self._order

    @property
    def tag(self):
        return self._tag

    def clear(self):
        self._dataset.clear()
        self._feature_keys.clear()

    def load(self, file_path, verbose=True, remake=False):
        if MPI.rank != 0:
            return

        if self.has_data:
            raise RuntimeError(
                'Cannot load dataset, since this dataset already has data.')

        # validate compatibility between my structures and loaded dataset
        ndarray = np.load(file_path)
        assert list(ndarray['elemental_composition']) \
               == self._elemental_composition
        assert list(ndarray['elements']) == self._elements
        assert ndarray['tag'].item() == self._tag
        assert len(ndarray[self._descriptors[0]]) == len(self)

        # validate lacking feature keys
        loaded_keys = list(ndarray['feature_keys'])
        lacking_keys = set(self._feature_keys) - set(loaded_keys)
        if lacking_keys:
            lacking_keys = '\n\t'.join(sorted(lacking_keys))
            if remake:
                if verbose:
                    pprint('Following feature keys are lacked in loaded'
                           f' {self.name} dataset.\n\t'
                           f'{lacking_keys}\n'
                           'Start to recalculate dataset from scratch.')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
                return
            else:
                raise ValueError(
                    'Following feature keys are lacked in loaded'
                    f' {self.name} dataset.\n'
                    'Please recalculate dataset from scratch.\n\t'
                    f'{lacking_keys}')

        # validate lacking descriptors
        lacking_descriptors = set(self._descriptors) - set(ndarray)
        if lacking_descriptors:
            lacking_descriptors = '\n\t'.join(sorted(lacking_descriptors))
            if remake:
                if verbose:
                    pprint('Following descriptors are lacked in loaded'
                           f' {self.name} dataset.\n\t'
                           f'{lacking_descriptors}\n'
                           'Start to recalculate dataset from scratch.')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
                return
            else:
                raise ValueError(
                    'Following descriptors are lacked in loaded'
                    f' {self.name} dataset.\n'
                    'Please recalculate dataset from scratch.\n\t'
                    f'{lacking_descriptors}')

        # load dataset as much as needed
        for i in range(self._order + 1):
            indices = np.array([loaded_keys.index(key)
                                for key in self._feature_keys])
            data = np.take(ndarray[self._descriptors[i]], indices, axis=2)
            self._dataset.append(data)

        if verbose:
            pprint(f'Successfully loaded & made needed {self.name} dataset'
                   f' from {file_path}')

    def save(self, file_path, verbose=True):
        if MPI.rank != 0:
            return

        if not self.has_data:
            raise RuntimeError(
                'Cannot save dataset,'
                ' since this dataset does not have any data.')

        data = {descriptor: data for descriptor, data
                in zip(self._descriptors, self._dataset)}
        info = {
            'elemental_composition': self._elemental_composition,
            'elements': self._elements,
            'feature_keys': self._feature_keys,
            'tag': self._tag,
            }
        np.savez(file_path, **data, **info)
        if verbose:
            pprint(f'Successfully saved {self.name} dataset to {file_path}.')

    @abstractmethod
    def generate_feature_keys(self, *args, **kwargs):
        return

    @abstractmethod
    def make(self, *args, **kwargs):
        pass

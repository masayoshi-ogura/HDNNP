# coding: utf-8

from abc import (ABC, abstractmethod)

import numpy as np

from hdnnpy.utils import (MPI, pprint)


class PropertyDatasetBase(ABC):
    PROPERTIES = []
    UNITS = []
    name = ''

    def __init__(self, order, structures):
        self._order = order
        self._properties = self.PROPERTIES[: order+1]
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._structures = structures
        self._tag = structures[0].info['tag']
        self._units = self.UNITS[: order+1]
        self._dataset = []

    def __getitem__(self, item):
        if isinstance(item, str):
            try:
                index = self._properties.index(item)
            except ValueError:
                raise KeyError(item) from None
            return self._dataset[index]
        else:
            return [data[item] for data in self._dataset]

    def __len__(self):
            return len(self._structures)

    @property
    def elemental_composition(self):
        return self._elemental_composition

    @property
    def elements(self):
        return self._elements

    @property
    def has_data(self):
        return len(self._dataset) != 0

    @property
    def order(self):
        return self._order

    @property
    def properties(self):
        return self._properties

    @property
    def tag(self):
        return self._tag

    @property
    def units(self):
        return self._units

    def clear(self):
        self._dataset.clear()

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
        assert len(ndarray[self._properties[0]]) == len(self)

        # validate lacking properties
        lacking_properties = set(self._properties) - set(ndarray)
        if lacking_properties:
            lacking_properties = '\n\t'.join(sorted(lacking_properties))
            if remake:
                if verbose:
                    pprint('Following properties are lacked in loaded'
                           f' {self.name} dataset.\n\t'
                           f'{lacking_properties}\n'
                           'Start to recalculate dataset from scratch.')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
            else:
                raise ValueError('Following properties are lacked in loaded'
                                 f' {self.name} dataset.\n'
                                 'Please recalculate dataset from scratch.\n\t'
                                 f'{lacking_properties}')

        # load dataset as much as needed
        for i in range(self._order + 1):
            self._dataset.append(ndarray[self._properties[i]])

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

        data = {property_: data for property_, data
                in zip(self._properties, self._dataset)}
        info = {
            'elemental_composition': self._elemental_composition,
            'elements': self._elements,
            'tag': self._tag,
            }
        np.savez(file_path, **data, **info)
        if verbose:
            pprint(f'Successfully saved {self.name} dataset to {file_path}.')

    @abstractmethod
    def make(self, *args, **kwargs):
        pass

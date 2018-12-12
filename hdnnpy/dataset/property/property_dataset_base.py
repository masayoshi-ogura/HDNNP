# -*- coding: utf-8 -*-

__all__ = [
    'PropertyDatasetBase',
    ]

from abc import (ABC,
                 abstractmethod,
                 )

from hdnnpy.utils import MPI


class PropertyDatasetBase(ABC):
    def __init__(self, order):
        self._order = order
        self._dataset = []
        self._elemental_composition = []
        self._elements = []
        self._properties = []
        self._tag = None

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
            return len(self._dataset[0])

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

    def clear(self):
        self._dataset.clear()
        self._elemental_composition.clear()
        self._elements.clear()
        self._tag = None


    @abstractmethod
    def load(self, *args, **kwargs):
        if MPI.rank == 0:
            pass

    @abstractmethod
    def make(self, *args, **kwargs):
        pass

    @abstractmethod
    def save(self, *args, **kwargs):
        if MPI.rank == 0:
            if not self.has_data:
                raise RuntimeError('This dataset does not have any data.')
            pass

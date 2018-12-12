# -*- coding: utf-8 -*-

__all__ = [
    'HDNNPDataset',
    ]

import numpy as np

from hdnnpy.dataset.descriptor import DESCRIPTOR_DATASET
from hdnnpy.dataset.property import PROPERTY_DATASET
from hdnnpy.utils import (MPI,
                          recv_chunk,
                          send_chunk,
                          )


RANDOMSTATE = np.random.get_state()


class HDNNPDataset(object):
    def __init__(self, descriptor, property_=None, order=0,
                 dataset=None, elemental_composition=None, elements=None,
                 total_size=0, tag=None):
        assert descriptor is not None
        self._descriptor = descriptor
        self._descriptor_dataset = DESCRIPTOR_DATASET[descriptor](order)

        self._property = property_
        if property_ is None:
            self._property_dataset = None
        else:
            self._property_dataset = PROPERTY_DATASET[property_](order)

        if dataset is None:
            dataset = []
        if elemental_composition is None:
            elemental_composition = []
        if elements is None:
            elements = []
        self._order = order
        self._dataset = dataset[:]
        self._elemental_composition = elemental_composition[:]
        self._elements = elements[:]
        self._total_size = total_size
        self._tag = tag

    def __getitem__(self, item):
        batches = [dataset[item] for dataset in self._dataset]
        if isinstance(item, slice):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        return self.partial_size

    @property
    def elemental_composition(self):
        return self._elemental_composition

    @property
    def elements(self):
        return self._elements

    @property
    def order(self):
        return self._order

    @property
    def partial_size(self):
        return len(self._dataset[0])

    @property
    def tag(self):
        return self._tag

    @property
    def total_size(self):
        return self._total_size

    @property
    def descriptor_dataset(self):
        return self._descriptor_dataset

    @property
    def property_dataset(self):
        return self._property_dataset

    def construct(self, preproc=None, shuffle=True):
        if MPI.rank == 0:
            # check compatibility and add info to myself
            self._check_dataset_compatibility()

            # apply pre-processing & append my dataset
            keys = self._descriptor_dataset.descriptors
            if preproc:
                input_dataset = preproc.decompose(
                    self._elemental_composition, self._elements,
                    *[self._descriptor_dataset[key] for key in keys])
            else:
                input_dataset = [self._descriptor_dataset[key] for key in keys]

            # merge dataset
            if self._property_dataset:
                keys = self._property_dataset.properties
                label_dataset = [self._property_dataset[key] for key in keys]
                self._dataset = input_dataset + label_dataset
            else:
                self._dataset = input_dataset

            # shuffle dataset
            if shuffle:
                self._shuffle()

            # delete duplicated datasets
            self._descriptor_dataset.clear()
            if self._property_dataset:
                self._property_dataset.clear()

    def scatter(self, root=0, max_buf_len=256 * 1024 * 1024):
        assert 0 <= root < MPI.size
        MPI.comm.Barrier()

        if MPI.rank == root:
            mine = None
            n_total_samples = self._total_size
            n_sub_samples = (n_total_samples+MPI.size-1) // MPI.size

            for i in range(MPI.size):
                b = n_total_samples * i // MPI.size
                e = b + n_sub_samples
                slc = slice(b, e, None)

                if i == root:
                    dataset = [data[slc] for data in self._dataset]
                    mine = [
                        self._descriptor, self._property, self._order,
                        dataset, self._elemental_composition, self._elements,
                        self._total_size, self._tag]

                else:
                    dataset = [data[slc] for data in self._dataset]
                    send = [
                        self._descriptor, self._property, self._order,
                        dataset, self._elemental_composition, self._elements,
                        self._total_size, self._tag]
                    send_chunk(send, dest=i, max_buf_len=max_buf_len)
            assert mine is not None
            self.__init__(*mine)

        else:
            recv = recv_chunk(source=root, max_buf_len=max_buf_len)
            assert recv is not None
            self.__init__(*recv)

    def take(self, index):
        dataset = [data[index] for data in self._dataset]
        new_dataset = self.__class__(
            self._descriptor, self._property, self._order,
            dataset, self._elemental_composition, self._elements,
            self._total_size, self._tag)
        return new_dataset

    def _check_dataset_compatibility(self):
        if not self._descriptor_dataset.has_data:
            raise ValueError('''
Cannot construct HDNNP dataset,
  because descriptor dataset does not have any data.
Use `this.descriptor_dataset.make() or load()`
  before constructing HDNNP dataset.
''')

        if self._property_dataset is None:
            self._elemental_composition = (
                self._descriptor_dataset.elemental_composition[:])
            self._elements = self._descriptor_dataset.elements[:]
            self._tag = self._descriptor_dataset.tag
            self._total_size = len(self._descriptor_dataset)
            return

        elif not self._property_dataset.has_data:
            raise ValueError('''
Cannot construct HDNNP dataset,
  because property dataset does not have any data.
Use `this.property_dataset.make() or load()`
  before constructing HDNNP dataset.
''')

        try:
            assert len(self._descriptor_dataset) \
                   == len(self._property_dataset)
            assert self._descriptor_dataset.elemental_composition \
                   == self._property_dataset.elemental_composition
            assert self._descriptor_dataset.elements \
                   == self._property_dataset.elements
            assert self._descriptor_dataset.order \
                   == self._property_dataset.order
            assert self._descriptor_dataset.tag \
                   == self._property_dataset.tag
        except AssertionError:
            d = self._descriptor_dataset
            p = self._property_dataset
            raise ValueError(f'''
Cannot construct HDNNP dataset, because the given descriptor dataset
  and property dataset is not compatible.
Both datasets must be the same for the following attributes.

    `__len__`, `elemental_composition`, `elements`, `order`, `tag`
    
Descriptor dataset has following values:
    `__len__` = {len(d)}
    `elemental_composition` = {d.elemental_composition}
    `elements` = {d.elements}
    `order` = {d.order}
    `tag` = {d.tag}
    
Property dataset has following values:
    `__len__` = {len(p)}
    `elemental_composition` = {p.elemental_composition}
    `elements` = {p.elements}
    `order` = {p.order}
    `tag` = {p.tag}
''')
        else:
            self._elemental_composition = (
                self._descriptor_dataset.elemental_composition[:])
            self._elements = self._descriptor_dataset.elements[:]
            self._tag = self._descriptor_dataset.tag
            self._total_size = len(self._descriptor_dataset)

    def _shuffle(self):
        for data in self._dataset:
            np.random.set_state(RANDOMSTATE)
            np.random.shuffle(data)

# coding: utf-8

import numpy as np

from hdnnpy.utils import (MPI, recv_chunk, send_chunk)


RANDOMSTATE = np.random.get_state()


class HDNNPDataset(object):
    def __init__(self, descriptor, property_, dataset=None):
        if dataset is None:
            dataset = []
        self._descriptor = descriptor
        self._property = property_
        self._dataset = dataset[:]

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
        return self._descriptor.elemental_composition

    @property
    def elements(self):
        return self._descriptor.elements

    @property
    def order(self):
        return self._descriptor.order

    @property
    def partial_size(self):
        return len(self._dataset[0])

    @property
    def tag(self):
        return self._descriptor.tag

    @property
    def total_size(self):
        return len(self._descriptor)

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def property(self):
        return self._property

    def construct(self, all_elements,
                  preprocesses=None, shuffle=True, verbose=True):
        if preprocesses is None:
            preprocesses = []
        if MPI.rank != 0:
            return

        # check compatibility and add info to myself
        self._check_dataset_compatibility()

        # apply pre-processing & append my dataset
        inputs = [self._descriptor[key]
                  for key in self._descriptor.descriptors]
        if all_elements != self._descriptor.elements:
            old_feature_keys = self._descriptor.feature_keys
            new_feature_keys = (
                self._descriptor.generate_feature_keys(all_elements))
            inputs = self._expand_feature_dims(
                inputs, old_feature_keys, new_feature_keys)
        for preprocess in preprocesses:
            inputs = preprocess.apply(
                inputs, self.elemental_composition, verbose=verbose)

        # merge dataset
        if self._property.has_data:
            labels = [self._property[key] for key in self._property.properties]
            self._dataset = inputs + labels
        else:
            self._dataset = inputs

        # shuffle dataset
        if shuffle:
            self._shuffle()

        # delete original datasets
        self._descriptor.clear()
        if self._property.has_data:
            self._property.clear()

    def scatter(self, root=0, max_buf_len=256 * 1024 * 1024):
        assert 0 <= root < MPI.size

        if MPI.rank == root:
            mine = None
            n_total_samples = self.total_size
            n_sub_samples = (n_total_samples+MPI.size-1) // MPI.size

            for i in range(MPI.size):
                b = n_total_samples * i // MPI.size
                e = b + n_sub_samples
                slc = slice(b, e, None)

                if i == root:
                    dataset = [data[slc] for data in self._dataset]
                    mine = [self._descriptor, self._property, dataset]

                else:
                    dataset = [data[slc] for data in self._dataset]
                    send = [self._descriptor, self._property, dataset]
                    send_chunk(send, dest=i, max_buf_len=max_buf_len)
            assert mine is not None
            self.__init__(*mine)

        else:
            recv = recv_chunk(source=root, max_buf_len=max_buf_len)
            assert recv is not None
            self.__init__(*recv)

    def take(self, index):
        dataset = [data[index] for data in self._dataset]
        new_dataset = self.__class__(self._descriptor, self._property, dataset)
        return new_dataset

    def _check_dataset_compatibility(self):
        if not self._descriptor.has_data:
            raise RuntimeError('Cannot construct HDNNP dataset, because'
                               ' descriptor dataset does not have any data.\n'
                               'Use `descriptor.make() or load()` before'
                               ' constructing HDNNP dataset.\n')

        elif not self._property.has_data:
            return

        else:
            assert len(self._descriptor) == len(self._property)
            assert self._descriptor.elemental_composition \
                   == self._property.elemental_composition
            assert self._descriptor.elements == self._property.elements
            assert self._descriptor.order == self._property.order
            assert self._descriptor.tag == self._property.tag

    @staticmethod
    def _expand_feature_dims(inputs, old_feature_keys, new_feature_keys):
        n_pad = len(new_feature_keys) - len(old_feature_keys)
        idx_pad = len(old_feature_keys)
        sort_indices = []
        for key in new_feature_keys:
            if key in old_feature_keys:
                sort_indices.append(old_feature_keys.index(key))
            else:
                sort_indices.append(idx_pad)
                idx_pad += 1
        sort_indices = np.array(sort_indices)

        for i, data in enumerate(inputs):
            n_pad = [(0, n_pad) if i == 2 else (0, 0)
                     for i in range(data.ndim)]
            np.pad(data, n_pad, 'constant')
            inputs[i] = data[:, :, sort_indices]
        return inputs

    def _shuffle(self):
        for data in self._dataset:
            np.random.set_state(RANDOMSTATE)
            np.random.shuffle(data)

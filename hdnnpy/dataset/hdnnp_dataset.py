# coding: utf-8

"""Combine and preprocess descriptor and property dataset."""

import numpy as np

from hdnnpy.utils import (MPI, recv_chunk, send_chunk)


RANDOMSTATE = np.random.get_state()


class HDNNPDataset(object):
    """Combine and preprocess descriptor and property dataset."""
    def __init__(self, descriptor, property_, dataset=None):
        """
        | It is desirable that the type of descriptor and property used
          for HDNNP is fixed at initialization.
        | Also, an instance itself does not have any dataset at
          initialization and you need to execute :meth:`construct`.
        | If ``dataset`` is given it will be an instance's own dataset.

        Args:
            descriptor (DescriptorDatasetBase):
                Descriptor instance you want to use as HDNNP input.
            property\_ (PropertyDatasetBase):
                Property instance you want to use as HDNNP label.
            dataset (dict [~numpy.ndarray], optional):
                If specified, dataset will be initialized with this.
        """
        if dataset is None:
            dataset = {}
        self._descriptor = descriptor
        self._property = property_
        self._dataset = dataset.copy()
        self._n_input = (dataset['inputs/0'].shape[-1]
                         if 'inputs/0' in dataset else None)
        self._n_label = (dataset['labels/0'].shape[-1]
                         if 'labels/0' in dataset else None)

    def __getitem__(self, item):
        """Return indexed or sliced dataset as dict data."""
        batches = {key: data[item]
                   for key, data in self._dataset.items()}
        if isinstance(item, slice):
            length = len(list(batches.values())[0])
            return [{key: batch[i] for key, batch in batches.items()}
                    for i in range(length)]
        else:
            return batches

    def __len__(self):
        """Redicect to :attr:`partial_size`"""
        return self.partial_size

    @property
    def descriptor(self):
        """DescriptorDatasetBase: Descriptor dataset instance."""
        return self._descriptor

    @property
    def elemental_composition(self):
        """list [str]: Elemental composition of the dataset."""
        return self._descriptor.elemental_composition

    @property
    def elements(self):
        """list [str]: Elements of the dataset."""
        return self._descriptor.elements

    @property
    def n_input(self):
        """int: Number of dimensions of input data."""
        return self._n_input or self._descriptor.n_feature

    @property
    def n_label(self):
        """int: Number of dimensions of label data."""
        return self._n_label or self._property.n_property

    @property
    def partial_size(self):
        """int: Number of data after scattered by MPI communication."""
        return len(list(self._dataset.values())[0])

    @property
    def tag(self):
        """str: Unique tag of the dataset.

        Usually, it is a form like ``<any prefix> <chemical formula>``.
        (ex. ``CrystalGa2N2``)
        """
        return self._descriptor.tag

    @property
    def total_size(self):
        """int: Number of data before scattered by MPI communication."""
        return len(self._descriptor)

    @property
    def property(self):
        """PropertyDatasetBase: Property dataset instance."""
        return self._property

    def construct(self, all_elements=None, preprocesses=None,
                  shuffle=True, verbose=True):
        """Construct an instance's own dataset.

        This method does following steps:

        * Check compatibility between descriptor and property datasets.
        * Expand feature dimension of descriptor dataset according to
          ``all_elements`` and pre-process descriptor dataset in a
          given order and add to its own dataset.
        * Add property dataset to its own dataset.
        * Clear up the original data in descriptor and property dataset.
        * Shuffle the order of the data.

        Args:
            all_elements (list [str], optional):
                If specified, it expands feature dimensions of
                descriptor dataset according to this.
            preprocesses (list [PreprocessBase], optional):
                If specified, it pre-processes descriptor dataset in a
                given order.
            shuffle (bool, optional):
                If specified, it shuffles the order of the data.
            verbose (bool, optional):
                Print log to stdout.

        Raises:
            AssertionError:
                If descriptor and property datasets are incompatible.
        """
        if preprocesses is None:
            preprocesses = []

        # check compatibility between descriptor and property datasets
        assert len(self._descriptor) == len(self._property)
        assert self._descriptor.elemental_composition \
               == self._property.elemental_composition
        assert self._descriptor.elements == self._property.elements
        assert self._descriptor.tag == self._property.tag

        # add descriptor dataset and delete original data
        if self._descriptor.has_data:
            inputs = [self._descriptor[key]
                      for key in self._descriptor.descriptors]
            # expand along to feature dimension
            if all_elements != self._descriptor.elements:
                old_feature_keys = self._descriptor.feature_keys
                new_feature_keys = (
                    self._descriptor.generate_feature_keys(all_elements))
                inputs = self._expand_feature_dims(
                    inputs, old_feature_keys, new_feature_keys)
            # pre-process descriptor dataset
            for preprocess in preprocesses:
                inputs = preprocess.apply(
                    inputs, self.elemental_composition, verbose=verbose)
            self._dataset.update(
                {f'inputs/{i}': data for i, data in enumerate(inputs)})
            self._descriptor.clear()
            self._n_input = inputs[0].shape[-1]

        # add property dataset and delete original data
        if self._property.has_data:
            labels = [self._property[key] for key in self._property.properties]
            self._dataset.update(
                {f'labels/{i}': data for i, data in enumerate(labels)})
            self._property.clear()
            self._n_label = labels[0].shape[-1]

        # shuffle dataset
        if shuffle:
            self._shuffle()

    def scatter(self, max_buf_len=256 * 1024 * 1024):
        """Scatter dataset by MPI communication.

        Each instance is re-initialized with received dataset.

        Args:
            max_buf_len (int, optional):
                Each data is divided into chunks of this size at
                maximum.
        """
        if MPI.rank == 0:
            new_dataset = {}
            MPI.comm.bcast(len(self._dataset), root=0)
            while self._dataset:
                key, data = self._dataset.popitem()
                for i, send_data in enumerate(np.array_split(data, MPI.size)):
                    if i == 0:
                        new_dataset[key] = send_data
                    else:
                        MPI.comm.send(key, dest=i)
                        send_chunk(send_data, dest=i, max_buf_len=max_buf_len)
            self._dataset.update(new_dataset)

        else:
            self._dataset.clear()
            n_data = MPI.comm.bcast(None, root=0)
            for i in range(n_data):
                key = MPI.comm.recv(source=0)
                recv_data = recv_chunk(source=0, max_buf_len=max_buf_len)
                self._dataset[key] = recv_data

    def take(self, index):
        """Return copied object that has sliced dataset.

        Args:
            index (int or slice):
                Copied object has dataset indexed or sliced by this.
        """
        dataset = {key: data[index] for key, data in self._dataset.items()}
        new_dataset = self.__class__(self._descriptor, self._property, dataset)
        return new_dataset

    @staticmethod
    def _expand_feature_dims(inputs, old_feature_keys, new_feature_keys):
        """Expand feature dimension of input dataset according to
        ``all_elements``."""
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
            pad_width = [(0, n_pad) if i == 2 else (0, 0)
                         for i in range(data.ndim)]
            data = np.pad(data, pad_width, 'constant')
            inputs[i] = data[:, :, sort_indices]
        return inputs

    def _shuffle(self):
        """Shuffle the order of the data."""
        for data in self._dataset.values():
            np.random.set_state(RANDOMSTATE)
            np.random.shuffle(data)

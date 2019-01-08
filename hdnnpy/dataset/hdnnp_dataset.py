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
            dataset (list [~numpy.ndarray], optional):
                If specified, dataset will be initialized with this.
        """
        if dataset is None:
            dataset = []
        self._descriptor = descriptor
        self._property = property_
        self._dataset = dataset[:]

    def __getitem__(self, item):
        """Return indexed or sliced dataset as tuple data."""
        batches = [dataset[item] for dataset in self._dataset]
        if isinstance(item, slice):
            length = len(batches[0])
            return [tuple([batch[i] for batch in batches])
                    for i in range(length)]
        else:
            return tuple(batches)

    def __len__(self):
        """Redicect to :attr:`partial_size`"""
        return self.partial_size

    @property
    def elemental_composition(self):
        """list [str]: Elemental composition of the dataset."""
        return self._descriptor.elemental_composition

    @property
    def elements(self):
        """list [str]: Elements of the dataset."""
        return self._descriptor.elements

    @property
    def order(self):
        """int: Derivative order of the dataset."""
        return self._descriptor.order

    @property
    def partial_size(self):
        """int: Number of data after scattered by MPI communication."""
        return len(self._dataset[0])

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
    def descriptor(self):
        """DescriptorDatasetBase: Descriptor dataset instance."""
        return self._descriptor

    @property
    def property(self):
        """PropertyDatasetBase: Property dataset instance."""
        return self._property

    def construct(self, all_elements=None, preprocesses=None,
                  shuffle=True, verbose=True):
        """Construct an instance's own dataset.

        This method does following steps:

        * Check compatibility between descriptor and property dataset.
        * Expand feature dimension of input dataset according to
          ``all_elements``.
        * Pre-process input dataset in a given order.
        * Merge input and label dataset to make its own dataset.
        * Shuffle the order of the data.
        * Clear up the original data in descriptor and property dataset.

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
            RuntimeError:
                If descriptor dataset does not have any data.
            AssertionError:
                If descriptor and property dataset are incompatible.
        """
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
        """Scatter dataset by MPI communication.

        Each instance is re-initialized with received dataset.

        Args:
            root (int, optional):
                Dataset is scattered from root MPI process.
            max_buf_len (int, optional):
                Each data is divided into chunks of this size at
                maximum.
        """
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
        """Return copied object that has sliced dataset.

        Args:
            index (int or slice):
                Copied object has dataset indexed or sliced by this.
        """
        dataset = [data[index] for data in self._dataset]
        new_dataset = self.__class__(self._descriptor, self._property, dataset)
        return new_dataset

    def _check_dataset_compatibility(self):
        """Check compatibility between descriptor and property dataset.
        """
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
        for data in self._dataset:
            np.random.set_state(RANDOMSTATE)
            np.random.shuffle(data)

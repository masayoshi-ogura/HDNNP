# coding: utf-8

"""Base class of atomic structure based descriptor dataset.

If you want to add new descriptor to extend HDNNP, inherits this base
class.
"""

from abc import (ABC, abstractmethod)

import numpy as np

from hdnnpy.utils import (MPI, pprint)


class DescriptorDatasetBase(ABC):
    """Base class of atomic structure based descriptor dataset."""
    DESCRIPTORS = []
    """list [str]: Names of descriptors for each derivative order."""
    name = ''
    """str: Name of this descriptor class."""

    def __init__(self, order, structures):
        """
        Common instance variables for descriptor datasets are
        initialized.

        Args:
            order (int): Derivative order of descriptor to calculate.
            structures (list [AtomicStructure]):
                Descriptors are calculated for these atomic structures.
        """
        self._order = order
        self._descriptors = self.DESCRIPTORS[: order+1]
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._structures = structures
        self._tag = structures[0].info['tag']
        self._dataset = []
        self._feature_keys = []

    def __getitem__(self, item):
        """Return descriptor data this instance has.

        If ``item`` is string, it returns corresponding descriptor.
        Available keys can be obtained by ``descriptors`` attribute.
        Otherwise, it returns a list of descriptor sliced by ``item``.
        """
        if isinstance(item, str):
            try:
                index = self._descriptors.index(item)
            except ValueError:
                raise KeyError(item) from None
            return self._dataset[index]
        else:
            return [data[item] for data in self._dataset]

    def __len__(self):
        """Number of atomic structures given at initialization."""
        return len(self._structures)

    @property
    def descriptors(self):
        """list [str]: Names of descriptors this instance have."""
        return self._descriptors

    @property
    def elemental_composition(self):
        """list [str]: Elemental composition of atomic structures given
        at initialization."""
        return self._elemental_composition

    @property
    def elements(self):
        """list [str]: Elements of atomic structures given at
        initialization."""
        return self._elements

    @property
    def feature_keys(self):
        """list [str]: Unique keys of feature dimension."""
        return self._feature_keys

    @property
    def has_data(self):
        """bool: True if success to load or make dataset,
        False otherwise."""
        has_data = len(self._dataset) > 0
        return MPI.bcast(has_data, root=0)

    @property
    def n_feature(self):
        """int: Length of feature dimension."""
        return len(self._feature_keys)

    @property
    def order(self):
        """int: Derivative order of descriptor to calculate."""
        return self._order

    @property
    def tag(self):
        """str: Unique tag of atomic structures given at
        initialization.

        Usually, it is a form like ``<any prefix> <chemical formula>``.
        (ex. ``CrystalGa2N2``)
        """
        return self._tag

    def clear(self):
        """Clear up instance variables to initial state."""
        self._dataset.clear()
        self._feature_keys.clear()

    def load(self, file_path, verbose=True, remake=False):
        """Load dataset from .npz format file.

        Only root MPI process load dataset.

        It validates following compatibility between loaded dataset and
        atomic structures given at initialization.

            * length of data
            * elemental composition
            * elements
            * tag

        It also validates that loaded dataset satisfies requirements.

            * feature keys
            * order

        Args:
            file_path (~pathlib.Path): File path to load dataset.
            verbose (bool, optional): Print log to stdout.
            remake (bool, optional): If loaded dataset is lacking in
                any feature key or any descriptor, recalculate dataset
                from scratch and overwrite it to ``file_path``.
                Otherwise, it raises ValueError.

        Raises:
            RuntimeError: If this instance already has data.
            AssertionError: If loaded dataset is incompatible with
                atomic structures given at initialization.
            ValueError: If loaded dataset is lacking in any feature key
                or any descriptor and ``remake=False``.
        """
        if self.has_data:
            raise RuntimeError('''
            Cannot load dataset, since this dataset already has data.
            ''')

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
                    pprint(f'''
                    Following feature keys are lacked in {file_path}.
                        {lacking_keys}
                    Start to recalculate dataset from scratch.
                    ''')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
                return
            else:
                raise ValueError(f'''
                Following feature keys are lacked in {file_path}.
                    {lacking_keys}
                Please recalculate dataset from scratch.
                ''')

        # validate lacking descriptors
        lacking_descriptors = set(self._descriptors) - set(ndarray)
        if lacking_descriptors:
            lacking_descriptors = '\n\t'.join(sorted(lacking_descriptors))
            if remake:
                if verbose:
                    pprint(f'''
                    Following descriptors are lacked in {file_path}.
                        {lacking_descriptors}
                    Start to recalculate dataset from scratch.
                    ''')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
                return
            else:
                raise ValueError(f'''
                Following descriptors are lacked in {file_path}.
                    {lacking_descriptors}
                Please recalculate dataset from scratch.
                ''')

        # load dataset as much as needed
        if MPI.rank == 0:
            for i in range(self._order + 1):
                indices = np.array([loaded_keys.index(key)
                                    for key in self._feature_keys])
                data = np.take(ndarray[self._descriptors[i]], indices, axis=2)
                self._dataset.append(data)

        if verbose:
            pprint(f'Successfully loaded & made needed {self.name} dataset'
                   f' from {file_path}')

    def save(self, file_path, verbose=True):
        """Save dataset to .npz format file.

        Only root MPI process save dataset.

        Args:
            file_path (~pathlib.Path): File path to save dataset.
            verbose (bool, optional): Print log to stdout.

        Raises:
            RuntimeError: If this instance do not have any data.
        """
        if not self.has_data:
            raise RuntimeError('''
            Cannot save dataset, since this dataset does not have any data.
            ''')

        if MPI.rank == 0:
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
        """Generate feature keys of current state.

        This is abstract method.
        Subclass of this base class have to override.

        Returns:
            list [str]: Unique keys of feature dimension.
        """
        return

    @abstractmethod
    def make(self, *args, **kwargs):
        """Calculate & retain descriptor dataset.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass

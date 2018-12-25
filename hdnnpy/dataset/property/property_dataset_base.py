# coding: utf-8

"""Base class of atomic structure based property dataset.

If you want to add new property to extend HDNNP, inherits this base
class.
"""

from abc import (ABC, abstractmethod)

import numpy as np

from hdnnpy.utils import (MPI, pprint)


class PropertyDatasetBase(ABC):
    """Base class of atomic structure based property dataset."""
    PROPERTIES = []
    """list [str]: Names of properties for each derivative order."""
    UNITS = []
    """list [str]: Units of properties for each derivative order."""
    name = ''
    """str: Name of this property class."""

    def __init__(self, order, structures):
        """Initialize property dataset base class.

        Common instance variables for property datasets are initialized.

        Args:
            order (int): Derivative order of property to calculate.
            structures (list [AtomicStructure]):
                Properties are calculated for these atomic structures.
        """
        self._order = order
        self._properties = self.PROPERTIES[: order+1]
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._structures = structures
        self._tag = structures[0].info['tag']
        self._units = self.UNITS[: order+1]
        self._dataset = []

    def __getitem__(self, item):
        """Return property data this instance has.

        If ``item`` is string, it returns corresponding property.
        Available keys can be obtained by ``properties`` attribute.
        Otherwise, it returns a list of property sliced by ``item``.
        """
        if isinstance(item, str):
            try:
                index = self._properties.index(item)
            except ValueError:
                raise KeyError(item) from None
            return self._dataset[index]
        else:
            return [data[item] for data in self._dataset]

    def __len__(self):
        """Number of atomic structures given at initialization."""
        return len(self._structures)

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
    def has_data(self):
        """bool: True if success to load or make dataset,
        False otherwise."""
        return len(self._dataset) != 0

    @property
    def order(self):
        """int: Derivative order of property to calculate."""
        return self._order

    @property
    def properties(self):
        """list [str]: Names of properties this instance have."""
        return self._properties

    @property
    def tag(self):
        """str: Unique tag of atomic structures given at
        initialization.

        Usually, it is a form like ``<any prefix> <chemical formula>``.
        (ex. ``CrystalGa2N2``)
        """
        return self._tag

    @property
    def units(self):
        """list [str]: Units of properties this instance have."""
        return self._units

    def clear(self):
        """Clear up instance variables to initial state."""
        self._dataset.clear()

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

            * order

        Args:
            file_path (~pathlib.Path): File path to load dataset.
            verbose (bool, optional): Print log to stdout.
            remake (bool, optional): If loaded dataset is lacking in
                any property, recalculate dataset from scratch and
                overwrite it to ``file_path``. Otherwise, it raises
                ValueError.

        Raises:
            RuntimeError: If this instance already has data.
            AssertionError: If loaded dataset is incompatible with
                atomic structures given at initialization.
            ValueError: If loaded dataset is lacking in any property and
                ``remake=False``.
        """
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
        """Save dataset to .npz format file.

        Only root MPI process save dataset.

        Args:
            file_path (~pathlib.Path): File path to save dataset.
            verbose (bool, optional): Print log to stdout.

        Raises:
            RuntimeError: If this instance do not have any data.
        """
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
        """Calculate & retain property dataset.

        This is abstract method.
        Subclass of this base class have to override.
        """
        pass

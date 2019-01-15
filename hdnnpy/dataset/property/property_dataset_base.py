# coding: utf-8

"""Base class of atomic structure based property dataset.

If you want to add new property to extend HDNNP, inherits this base
class.
"""

from abc import (ABC, abstractmethod)

import numpy as np
from tqdm import tqdm

from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class PropertyDatasetBase(ABC):
    """Base class of atomic structure based property dataset."""
    PROPERTIES = []
    """list [str]: Names of properties for each derivative order."""
    COEFFICIENTS = []
    """list [float]: Coefficient values of each properties."""
    UNITS = []
    """list [str]: Units of properties for each derivative order."""
    name = None
    """str: Name of this property class."""
    n_property = None
    """int: Number of dimensions of 0th property."""

    def __init__(self, order, structures):
        """
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
        self._length = len(structures)
        self._slices = [slice(i[0], i[-1]+1)
                        for i in np.array_split(range(self._length), MPI.size)]
        self._structures = structures[self._slices[MPI.rank]]
        self._tag = structures[0].info['tag']
        self._coefficients = self.COEFFICIENTS[: order+1]
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
        return self._length

    @property
    def coefficients(self):
        """list [float]: Coefficient values this instance have."""
        return self._coefficients

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
        return len(self._dataset) == self._order + 1

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
            AssertionError: If loaded dataset is incompatible with
                atomic structures given at initialization.
            ValueError: If loaded dataset is lacking in any property and
                ``remake=False``.
        """
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
            if verbose:
                lacking = ('\n'+' '*20).join(sorted(lacking_properties))
                pprint(f'''
                Following properties are lacked in {file_path}.
                    {lacking}
                ''')
            if remake:
                if verbose:
                    pprint('Start to recalculate dataset from scratch.')
                self.make(verbose=verbose)
                self.save(file_path, verbose=verbose)
                return
            else:
                raise ValueError('Please recalculate dataset from scratch.')

        # load dataset as much as needed
        if MPI.rank == 0:
            for i in range(self._order + 1):
                self._dataset.append(ndarray[self._properties[i]])

        if verbose:
            pprint(f'Successfully loaded & made needed {self.name} dataset'
                   f' from {file_path}')

    def make(self, verbose=True):
        """Calculate & retain property dataset

        | It calculates property dataset by data-parallel using MPI
          communication.
        | The calculated dataset is retained in only root MPI process.

        Each property values are divided by ``COEFFICIENTS`` which is
        unique to each property dataset class.

        Args:
            verbose (bool, optional): Print log to stdout.
        """
        dataset = []
        for structure in tqdm(self._structures,
                              ascii=True, desc=f'Process #{MPI.rank}',
                              leave=False, position=MPI.rank):
            dataset.append(self.calculate_properties(structure))

        for data_list, coefficient in zip(zip(*dataset), self._coefficients):
            shape = data_list[0].shape
            send_data = np.stack(data_list) / coefficient
            del data_list
            if MPI.rank == 0:
                recv_data = np.empty((self._length, *shape), dtype=np.float32)
                recv_data[self._slices[0]] = send_data
                del send_data
                for j in range(1, MPI.size):
                    recv_data[self._slices[j]] = recv_chunk(source=j)
                self._dataset.append(recv_data)
            else:
                send_chunk(send_data, dest=0)
                del send_data

        if verbose:
            pprint(f'Calculated {self.name} dataset.')

    def save(self, file_path, verbose=True):
        """Save dataset to .npz format file.

        Only root MPI process save dataset.

        Args:
            file_path (~pathlib.Path): File path to save dataset.
            verbose (bool, optional): Print log to stdout.

        Raises:
            RuntimeError: If this instance do not have any data.
        """
        if not MPI.comm.bcast(self.has_data, root=0):
            raise RuntimeError('''
            Cannot save dataset, since this dataset does not have any data.
            ''')

        if MPI.rank == 0:
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
    def calculate_properties(self, structure):
        """Calculate required properties for a structure data.

        This is abstract method.
        Subclass of this base class have to override.

        Args:
            structure (AtomicStructure):
                A structure data to calculate properties.

        Returns:
            list [~numpy.ndarray]: Calculated properties.
            The length is the same as ``order`` given at initialization.
        """
        return

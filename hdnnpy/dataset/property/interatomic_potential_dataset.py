# coding: utf-8

"""Interatomic potential dataset for property of HDNNP. """

import numpy as np
from tqdm import tqdm

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class InteratomicPotentialDataset(PropertyDatasetBase):
    """Interatomic potential dataset for property of HDNNP. """
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']
    """list [str]: Names of properties for each derivative order."""
    COEFFICIENTS = [1.0, -1.0]
    """list [float]: Coefficient values of each properties."""
    UNITS = ['eV/atom', 'eV/$\\AA$']
    """list [str]: Units of properties for each derivative order."""
    name = 'interatomic_potential'
    """str: Name of this property class."""

    def __init__(self, order, structures):
        """
        It accepts 0 or 3 for ``order``.

        Notes:
            Currently you cannot use order = 2 or 3, since it is not
            implemented.

        Args:
            order (int): passed to super class.
            structures (list [AtomicStructure]): passed to super class.
        """
        assert 0 <= order <= 3
        super().__init__(order, structures)

    def make(self, verbose=True):
        """Calculate & retain interatomic potential dataset.

        | It calculates required properties by data-parallel using MPI
          communication.
        | The calculated dataset is retained in only root MPI process.

        Args:
            verbose (bool, optional): Print log to stdout.
        """
        dataset = []
        for structure in tqdm(self._structures,
                              ascii=True, desc=f'Process #{MPI.rank}',
                              leave=False, position=MPI.rank):
            dataset.append(self._calculate_properties(structure))

        for i, data_list in enumerate(zip(*dataset)):
            n_atom = len(self._structures[0])
            shape = (1, *(3*n_atom,) * i)
            send_data = np.stack(data_list).reshape((-1, *shape))
            if MPI.rank == 0:
                recv_data = np.empty((self._length, *shape), dtype=np.float32)
                recv_data[self._slices[0]] = send_data
                for j in range(1, MPI.size):
                    recv_data[self._slices[j]] = recv_chunk(source=j)
                self._dataset.append(recv_data)
            else:
                send_chunk(send_data, dest=0)

        if verbose:
            pprint(f'Calculated {self.name} dataset.')

    def _calculate_properties(self, structure):
        """Main method of calculating interatomic potential dataset."""
        dataset = []
        if self._order >= 0:
            dataset.append(self._calculate_energy(structure)
                           / self._coefficients[0])
        if self._order >= 1:
            dataset.append(self._calculate_force(structure)
                           / self._coefficients[1])
        if self._order >= 2:
            dataset.append(self._calculate_harmonic(structure)
                           / self._coefficients[2])
        if self._order >= 3:
            dataset.append(self._calculate_third_order(structure)
                           / self._coefficients[3])
        return dataset

    @staticmethod
    def _calculate_energy(structure):
        """Calculate atomic energy."""
        return (structure.get_potential_energy() / len(structure)
                ).astype(np.float32)

    @staticmethod
    def _calculate_force(structure):
        """Calculate interatomic forces."""
        return structure.get_forces().astype(np.float32)

    @staticmethod
    def _calculate_harmonic(structure):
        """Calculate 2nd-order harmonic force constant."""
        raise NotImplementedError

    @staticmethod
    def _calculate_third_order(structure):
        """Calculate 3rd-order anharmonic force constant."""
        raise NotImplementedError

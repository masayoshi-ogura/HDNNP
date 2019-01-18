# coding: utf-8

"""Born effective charge dataset for property of HDNNP. """

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class BornEffectiveChargeDataset(PropertyDatasetBase):
    """Born effective charge dataset for property of HDNNP. """
    PROPERTIES = ['energy', 'born_z']
    """list [str]: Names of properties for each derivative order."""
    COEFFICIENTS = [1.0, 1.0]
    """list [float]: Coefficient values of each properties."""
    UNITS = ['eV/atom', 'e']
    """list [str]: Units of properties for each derivative order."""
    name = 'born_effective_charge'
    """str: Name of this property class."""

    def __init__(self, order, structures):
        """
        It accepts 1 for ``order``.
        
        Args:
            order (int): passed to super class.
            structures (list [AtomicStructure]): passed to super class.
        """
        assert order == 1
        super().__init__(order, structures)

    def make(self, verbose=True):
        """Calculate & retain Born effective charge dataset.
        | It calculates required properties by data-parallel using MPI
          communication.
        | The calculated dataset is retained in only root MPI process.
        Args:
            verbose (bool, optional): Print log to stdout.
        """
        n_sample = len(self._structures)
        n_atom = len(self._structures[0])
        slices = [slice(i[0], i[-1]+1)
                  for i in np.array_split(range(n_sample), MPI.size)]
        structures = self._structures[slices[MPI.rank]]

        for i, send_data in enumerate(self._calculate_properties(structures)):
            shape = (1, *(n_atom, 3) * i)
            send_data = send_data.reshape((-1,) + shape)
            if MPI.rank == 0:
                data = np.empty((n_sample,) + shape, dtype=np.float32)
                data[slices[0]] = send_data
                for j in range(1, MPI.size):
                    data[slices[j]] = recv_chunk(source=j)
                self._dataset.append(data)
            else:
                send_chunk(send_data, dest=0)

        if verbose:
            pprint(f'Calculated {self.name} dataset.')

    def _calculate_properties(self, structures):
        """Main method of calculating Born effective charge dataset."""
        if self._order >= 0:
            yield (self._calculate_energy(structures)
                   / self._coefficients[0])
        if self._order >= 1:
            yield (self._calculate_Born_effective_charge(structures)
                   / self._coefficients[1])


    @staticmethod
    def _calculate_energy(structures):
        """Calculate atomic energy."""
        return np.array([structure.get_potential_energy() / len(structure)
                         for structure in structures],
                        dtype=np.float32)

    @staticmethod
    def _calculate_Born_effective_charge(structures):
        """Calculate Born effective charge."""
        return np.array([structure.get_array("born_z")
                         for structure in structures],
                        dtype=np.float32)

# coding: utf-8

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class InteratomicPotentialDataset(PropertyDatasetBase):
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']
    UNITS = ['eV/atom', 'eV/$\\AA$']
    name = 'interatomic_potential'

    def __init__(self, order, structures):
        assert 0 <= order <= 3
        super().__init__(order, structures)

    def make(self, verbose=True):
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
        if self._order >= 0:
            yield self._calculate_energy(structures)
        if self._order >= 1:
            yield self._calculate_force(structures)
        if self._order >= 2:
            yield self._calculate_harmonic(structures)
        if self._order >= 3:
            yield self._calculate_third_order(structures)

    @staticmethod
    def _calculate_energy(structures):
        return np.array([structure.get_potential_energy() / len(structure)
                         for structure in structures],
                        dtype=np.float32)

    @staticmethod
    def _calculate_force(structures):
        return np.array([structure.get_forces()
                         for structure in structures],
                        dtype=np.float32)

    @staticmethod
    def _calculate_harmonic(structures):
        raise NotImplementedError

    @staticmethod
    def _calculate_third_order(structures):
        raise NotImplementedError

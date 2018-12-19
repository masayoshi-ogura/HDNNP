# coding: utf-8

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.utils import (MPI, pprint)


class InteratomicPotentialDataset(PropertyDatasetBase):
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']
    UNITS = ['eV/atom', 'eV/$\AA$']
    name = 'interatomic_potential'

    def __init__(self, order, structures):
        assert 0 <= order <= 3
        super().__init__(order, structures)

    def make(self, verbose=True):
        n_sample = len(self._structures)
        n_atom = len(self._structures[0])
        count = np.array([(n_sample + i) // MPI.size
                          for i in range(MPI.size)[::-1]],
                         dtype=np.int32)
        s = count[: MPI.rank].sum()
        e = count[: MPI.rank+1].sum()
        structures = self._structures[s:e]

        for i, send_data in enumerate(self._calculate_properties(structures)):
            if MPI.rank == 0:
                shape = (n_sample, 1, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                recv_data = (data, count * data[0].size)
                MPI.comm.Gatherv(send_data, recv_data, root=0)
                self._dataset.append(data)
            else:
                MPI.comm.Gatherv(send_data, None, root=0)

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

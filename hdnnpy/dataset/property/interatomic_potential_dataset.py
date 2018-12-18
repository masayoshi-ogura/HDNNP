# coding: utf-8

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class InteratomicPotentialDataset(PropertyDatasetBase):
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']
    UNITS = ['eV', 'eV/$\AA$']
    name = 'interatomic_potential'

    def __init__(self, order=0):
        assert 0 <= order <= 3
        super().__init__(order)
        self._properties = self.PROPERTIES[: order+1]
        self._units = self.UNITS[: order+1]

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            for i in range(self._order + 1):
                self._dataset.append(ndarray[self._properties[i]])
            self._elemental_composition = list(
                ndarray['elemental_composition'])
            self._elements = list(ndarray['elements'])
            self._tag = ndarray['tag'].item()
            if verbose:
                pprint(
                    f'Loaded interatomic potential dataset from {file_path}.')

    def make(self, structures, verbose=True):
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._tag = structures[0].info['tag']
        n_sample = len(structures)
        n_atom = len(structures[0])

        count = np.array([(n_sample + i) // MPI.size
                          for i in range(MPI.size)[::-1]], dtype=np.int32)
        s = count[: MPI.rank].sum()
        e = count[: MPI.rank+1].sum()
        structures = structures[s:e]

        for i, send_data in enumerate(self._calculate_properties(structures)):
            recv_data = None
            if MPI.rank == 0:
                shape = (n_sample, 1, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                recv_data = (data, count * data[0].size)
                MPI.comm.Gatherv(send_data, recv_data, root=0)
                self._dataset.append(data)
            else:
                MPI.comm.Gatherv(send_data, recv_data, root=0)

        if verbose:
            pprint('Calculated interatomic potential dataset.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            if not self.has_data:
                raise RuntimeError('This dataset does not have any data.')

            data = {property_: data for property_, data
                    in zip(self._properties, self._dataset)}
            info = {
                'elemental_composition': self._elemental_composition,
                'elements': self._elements,
                'tag': self._tag,
                }
            np.savez(file_path, **data, **info)
            if verbose:
                pprint(f'Saved interatomic potential dataset to {file_path}.')

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
        return np.array([structure.get_potential_energy()
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

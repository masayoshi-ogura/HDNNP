# -*- coding: utf-8 -*-

__all__ = [
    'InteratomicPotentialDataset',
    ]

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase
from hdnnpy.settings import stg
from hdnnpy.utils import pprint


class InteratomicPotentialDataset(PropertyDatasetBase):
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']

    def __init__(self, order=0):
        assert 0 <= order <= 3
        super(InteratomicPotentialDataset, self).__init__(order)
        self._properties = self.PROPERTIES[: order+1]

    def make(self, atoms, verbose=True):
        self._elemental_composition = atoms[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._tag = atoms[0].info['tag']
        n_sample = len(atoms)
        n_atom = len(atoms[0])

        count = np.array([(n_sample + i) // stg.mpi.size
                          for i in range(stg.mpi.size)[::-1]], dtype=np.int32)
        s = count[: stg.mpi.rank].sum()
        e = count[: stg.mpi.rank+1].sum()
        atoms = atoms[s:e]

        for i, send_data in enumerate(self._calculate_properties(atoms)):
            recv_data = None
            if stg.mpi.rank == 0:
                shape = (n_sample, 1, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                recv_data = (data, count * data[0].size)
                stg.mpi.comm.Gatherv(send_data, recv_data, root=0)
                self._dataset.append(data)
            else:
                stg.mpi.comm.Gatherv(send_data, recv_data, root=0)

        if verbose:
            pprint('Calculated interatomic potential dataset.')

    def load(self, file_path, verbose=True):
        if stg.mpi.rank == 0:
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

    def save(self, file_path, verbose=True):
        if stg.mpi.rank == 0:
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

    def _calculate_properties(self, atoms):
        if self._order >= 0:
            yield self._calculate_energy(atoms)
        if self._order >= 1:
            yield self._calculate_force(atoms)
        if self._order >= 2:
            yield self._calculate_harmonic(atoms)
        if self._order >= 3:
            yield self._calculate_third_order(atoms)

    @staticmethod
    def _calculate_energy(atoms):
        return np.array([data.get_potential_energy() for data in atoms],
                        dtype=np.float32)

    @staticmethod
    def _calculate_force(atoms):
        return np.array([data.get_forces() for data in atoms],
                        dtype=np.float32)

    @staticmethod
    def _calculate_harmonic(atoms):
        raise NotImplementedError

    @staticmethod
    def _calculate_third_order(atoms):
        raise NotImplementedError

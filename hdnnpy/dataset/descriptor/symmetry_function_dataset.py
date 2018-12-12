# -*- coding: utf-8 -*-

__all__ = [
    'SymmetryFunctionDataset',
    ]

from collections import defaultdict
from itertools import (combinations,
                       combinations_with_replacement,
                       product,
                       )

import numpy as np

from hdnnpy.dataset.atomic_structure import neighbour_info
from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class SymmetryFunctionDataset(DescriptorDatasetBase):
    DESCRIPTORS = ['sym_func', 'derivative']

    def __init__(self, order=0):
        assert 0 <= order <= 1
        super(SymmetryFunctionDataset, self).__init__(order)
        self._descriptors = self.DESCRIPTORS[: order+1]

    def make(self, atoms, params, verbose=True):
        self._elemental_composition = atoms[0].get_chemical_symbols()
        self._elements = sorted(set(self._elemental_composition))
        self._params = params.copy()
        self._tag = atoms[0].info['tag']
        n_sample = len(atoms)
        n_atom = len(atoms[0])

        count = np.array([(n_sample + i) // MPI.size
                          for i in range(MPI.size)[::-1]], dtype=np.int32)
        s = count[: MPI.rank].sum()
        e = count[: MPI.rank+1].sum()
        atoms = atoms[s:e]

        for i, send_data in enumerate(self._calculate_descriptors(atoms)):
            recv_data = None
            if MPI.rank == 0:
                shape = (n_sample, n_atom, self._n_feature, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                recv_data = (data, count * data[0].size)
                MPI.comm.Gatherv(send_data, recv_data, root=0)
                self._dataset.append(data)
            else:
                MPI.comm.Gatherv(send_data, recv_data, root=0)

        if verbose:
            pprint('Calculated symmetry function dataset.')

    def dump_params(self, file_path):
        pass  # TODO

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            for i in range(self._order + 1):
                self._dataset.append(ndarray[self._descriptors[i]])
            self._elemental_composition = list(
                ndarray['elemental_composition'])
            self._elements = list(ndarray['elements'])
            self._n_feature = ndarray['n_feature'].item()
            self._params = ndarray['params'].item()
            self._tag = ndarray['tag'].item()
            if verbose:
                pprint(f'Loaded symmetry function dataset from {file_path}.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            if not self.has_data:
                raise RuntimeError('This dataset does not have any data.')

            data = {property_: data for property_, data
                    in zip(self._descriptors, self._dataset)}
            info = {
                'elemental_composition': self._elemental_composition,
                'elements': self._elements,
                'n_feature': self._n_feature,
                'params': self._params,
                'tag': self._tag,
                }
            np.savez(file_path, **data, **info)
            if verbose:
                pprint(f'Saved symmetry function dataset to {file_path}.')

    def _calculate_descriptors(self, atoms):
        twobody_combinations = self._elements
        threebody_combinations = list(
            combinations_with_replacement(self._elements, 2))
        n_type1 = len(list(product(twobody_combinations,
                                   self._params['Rc'])))
        n_type2 = len(list(product(twobody_combinations,
                                   self._params['Rc'],
                                   self._params['eta'],
                                   self._params['Rs'])))
        n_type4 = len(list(product(threebody_combinations,
                                   self._params['Rc'],
                                   self._params['eta'],
                                   self._params['lambda_'],
                                   self._params['zeta'])))
        n_twobody = len(twobody_combinations)
        n_threebody = len(threebody_combinations)

        n_sample = len(atoms)
        n_atom = len(atoms[0])
        self._n_feature = n_type1 + n_type2 + n_type4

        idx_feature = defaultdict(dict)
        for index, (elem1, elem2) in enumerate(threebody_combinations):
            idx_feature[elem1][elem2] = idx_feature[elem2][elem1] = index

        if self._order == 0:
            G = np.zeros(
                (n_sample, n_atom, self._n_feature),
                dtype=np.float32)
            for i, at in enumerate(atoms):
                for formula, slc, param_set in self._iterate_params(
                        n_twobody, n_threebody):
                    eval(f'self._calculate_{formula}')(
                        G[i, :, slc], idx_feature, at, *param_set)
            yield G

        elif self._order == 1:
            G = np.zeros(
                (n_sample, n_atom, self._n_feature),
                dtype=np.float32)
            dGdr = np.zeros(
                (n_sample, n_atom, self._n_feature, n_atom, 3),
                dtype=np.float32)
            for i, at in enumerate(atoms):
                for formula, slc, param_set in self._iterate_params(
                        n_twobody, n_threebody):
                    eval(f'self._calculate_{formula}_with_derivative')(
                        G[i, :, slc], dGdr[i, :, slc],
                        idx_feature, at, *param_set)
            yield G
            yield dGdr

    def _iterate_params(self, n_twobody, n_threebody):
        current = 0
        for param_set in product(self._params['Rc']):
            yield 'type1', slice(current, current + n_twobody), param_set
            current += n_twobody
        for param_set in product(self._params['Rc'], self._params['eta'],
                                 self._params['Rs']):
            yield 'type2', slice(current, current + n_twobody), param_set
            current += n_twobody
        for param_set in product(self._params['Rc'], self._params['eta'],
                                 self._params['lambda_'],
                                 self._params['zeta']):
            yield 'type4', slice(current, current + n_threebody), param_set
            current += n_threebody

    def _calculate_type1(self, G, idx_feature, atoms, Rc):
        idx_feature = idx_feature[self._elements[0]]
        for i, idx_neigh, R, tanh, _, _, _ in neighbour_info(atoms, Rc):
            g = tanh**3
            for elem_j in self._elements:
                G[i, idx_feature[elem_j]] = g[idx_neigh[elem_j]].sum()

    def _calculate_type1_with_derivative(
            self, G, dGdr, idx_feature, atoms, Rc):
        idx_feature = idx_feature[self._elements[0]]
        for i, idx_neigh, R, tanh, dR, _, _ in neighbour_info(atoms, Rc):
            g = tanh**3
            dgdr = - 3.0 / Rc * ((1.0-tanh**2)*tanh**2)[:, None] * dR
            for elem_j in self._elements:
                G[i, idx_feature[elem_j]] = g[idx_neigh[elem_j]].sum()
            for j, elem_j in enumerate(self._elemental_composition):
                dGdr[i, idx_feature[elem_j], j] = (dgdr
                                                   .take(idx_neigh[j], 0)
                                                   .sum(0))

    def _calculate_type2(self, G, idx_feature, atoms, Rc, eta, Rs):
        idx_feature = idx_feature[self._elements[0]]
        for i, idx_neigh, R, tanh, _, _, _ in neighbour_info(atoms, Rc):
            g = np.exp(-eta*(R-Rs)**2) * tanh**3
            for elem_j in self._elements:
                G[i, idx_feature[elem_j]] = g[idx_neigh[elem_j]].sum()

    def _calculate_type2_with_derivative(
            self, G, dGdr, idx_feature, atoms, Rc, eta, Rs):
        idx_feature = idx_feature[self._elements[0]]
        for i, idx_neigh, R, tanh, dR, _, _ in neighbour_info(atoms, Rc):
            g = np.exp(-eta*(R-Rs)**2) * tanh**3
            dgdr = (np.exp(-eta*(R-Rs)**2)
                    * tanh**2
                    * (-2.0*eta*(R-Rs)*tanh + 3.0/Rc*(tanh**2-1.0))
                    )[:, None] * dR
            for elem_j in self._elements:
                G[i, idx_feature[elem_j]] = g[idx_neigh[elem_j]].sum()
            for j, elem_j in enumerate(self._elemental_composition):
                dGdr[i, idx_feature[elem_j], j] = (dgdr
                                                   .take(idx_neigh[j], 0)
                                                   .sum(0))

    def _calculate_type4(self, G, idx_feature, atoms, Rc, eta, lambda_, zeta):
        for i, idx_neigh, R, tanh, _, cos, _ in neighbour_info(atoms, Rc):
            ang = 1.0 + lambda_*cos
            ang[np.eye(len(R), dtype=bool)] = 0
            rad = np.exp(-eta*R**2) * tanh**3

            g = 2.0**(1-zeta) * ang**zeta * rad[:, None] * rad[None, :]

            for elem_j in self._elements:
                G[i, idx_feature[elem_j][elem_j]] = (
                        g
                        .take(idx_neigh[elem_j], 0)
                        .take(idx_neigh[elem_j], 1)
                        .sum()
                        / 2.0)
            for elem_j, elem_k in combinations(self._elements, 2):
                G[i, idx_feature[elem_j][elem_k]] = (
                        g
                        .take(idx_neigh[elem_j], 0)
                        .take(idx_neigh[elem_k], 1)
                        .sum())

    def _calculate_type4_with_derivative(
            self, G, dGdr, idx_feature, atoms, Rc, eta, lambda_, zeta):
        for i, idx_neigh, R, tanh, dR, cos, dcos in neighbour_info(atoms, Rc):
            ang = 1.0 + lambda_*cos
            ang[np.eye(len(R), dtype=bool)] = 0
            rad1 = np.exp(-eta*R**2) * tanh**3
            rad2 = (np.exp(-eta*R**2)
                    * tanh**2
                    * (-2.0*eta*R*tanh + 3.0/Rc*(tanh**2-1.0))
                    )

            g = 2.0**(1-zeta) * ang**zeta * rad1[:, None] * rad1[None, :]
            dgdr_radial_part = (2.0**(1-zeta)
                                * ang[:, :, None]**zeta
                                * rad2[:, None, None]
                                * rad1[None, :, None]
                                * dR[:, None, :])
            dgdr_angular_part = (zeta * lambda_
                                 * 2.0**(1-zeta)
                                 * ang[:, :, None]**(zeta-1)
                                 * rad1[:, None, None]
                                 * rad1[None, :, None]
                                 * dcos)
            dgdr = dgdr_radial_part + dgdr_angular_part

            for elem_j in self._elements:
                G[i, idx_feature[elem_j][elem_j]] = (
                        g
                        .take(idx_neigh[elem_j], 0)
                        .take(idx_neigh[elem_j], 1)
                        .sum()
                        / 2.0)
            for elem_j, elem_k in combinations(self._elements, 2):
                G[i, idx_feature[elem_j][elem_k]] = (
                        g
                        .take(idx_neigh[elem_j], 0)
                        .take(idx_neigh[elem_k], 1)
                        .sum())
            for (j, elem_j), elem_k in product(
                    enumerate(self._elemental_composition), self._elements):
                dGdr[i, idx_feature[elem_j][elem_k], j] = (
                        dgdr
                        .take(idx_neigh[j], 0)
                        .take(idx_neigh[elem_k], 1)
                        .sum((0, 1)))

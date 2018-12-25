# coding: utf-8

from itertools import (chain, combinations_with_replacement)

import numpy as np
from scipy.linalg import block_diag
from tqdm import tqdm

from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)
from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class SymmetryFunctionDataset(DescriptorDatasetBase):
    DESCRIPTORS = ['sym_func', 'derivative']
    name = 'symmetry_function'

    def __init__(self, order, structures, **func_param_map):
        assert 0 <= order <= 1
        assert func_param_map
        super().__init__(order, structures)
        self._func_param_map = func_param_map.copy()
        self._feature_keys = self.generate_feature_keys(self._elements)

    @property
    def function_names(self):
        return list(self._func_param_map.keys())

    @property
    def params(self):
        return self._func_param_map

    def generate_feature_keys(self, elements):
        feature_keys = []
        for function_name, params_set in self._func_param_map.items():
            for params in params_set:
                param_key = '/'.join(map(str, params))
                if function_name in ['type1', 'type2']:
                    for element_key in elements:
                        key = ':'.join([function_name, param_key, element_key])
                        feature_keys.append(key)

                elif function_name in ['type4']:
                    for combo in combinations_with_replacement(elements, 2):
                        element_key = '/'.join(combo)
                        key = ':'.join([function_name, param_key, element_key])
                        feature_keys.append(key)
        return feature_keys

    def make(self, verbose=True):
        n_sample = len(self._structures)
        n_atom = len(self._structures[0])
        slices = [slice(i[0], i[-1]+1)
                  for i in np.array_split(range(n_sample), MPI.size)]
        structures = self._structures[slices[MPI.rank]]

        for i, send_data in enumerate(self._calculate_descriptors(structures)):
            if MPI.rank == 0:
                shape = (n_sample, n_atom, self.n_feature, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                data[slices[0]] = send_data
                for j in range(1, MPI.size):
                    data[slices[j]] = recv_chunk(source=j)
                self._dataset.append(data)
            else:
                send_chunk(send_data, dest=0)

        if verbose:
            pprint(f'Calculated {self.name} dataset.')

    def _calculate_descriptors(self, structures):
        n_sample = len(structures)
        n_atom = len(structures[0])

        functions = []
        for function_name, params_set in self._func_param_map.items():
            for params in params_set:
                functions.append(eval(
                    f'self._make_symmetry_function_{function_name}')(*params))

        if self._order >= 0:
            G = np.zeros((n_sample, n_atom, self.n_feature),
                         dtype=np.float32)
            for i, structure in enumerate(tqdm(
                    structures, ascii=True, desc=f'Process #{MPI.rank}',
                    leave=False, position=MPI.rank)):
                for j, g in enumerate(zip(
                        *[func[0](structure) for func in functions])):
                    G[i, j] = list(chain.from_iterable(g))
                if self._order == 0:
                    structure.clear_cache()
            yield G

        if self._order >= 1:
            dGdr = np.zeros((n_sample, n_atom, self.n_feature, n_atom, 3),
                            dtype=np.float32)
            for i, structure in enumerate(tqdm(
                    structures, ascii=True, desc=f'Process #{MPI.rank}',
                    leave=False, position=MPI.rank)):
                for j, dgdr in enumerate(zip(
                        *[func[1](structure) for func in functions])):
                    dGdr[i, j] = list(chain.from_iterable(dgdr))
                if self._order == 1:
                    structure.clear_cache()
            yield dGdr

    def _make_symmetry_function_type1(self, Rc):
        def symmetry_function_type1(structure):
            for R, neigh2elem in structure.get_neighbor_info(
                    Rc, ['distance', 'neigh2elem']):
                g = np.tanh(1.0 - R/Rc) ** 3
                yield [np.sum(g_e) for g_e in np.split(g, neigh2elem[1:])]

        def diff_symmetry_function_type1(structure):
            n_atom = len(structure)
            for R, dRdr, neigh2j, j2elem in structure.get_neighbor_info(
                    Rc, ['distance', 'diff_distance', 'neigh2j', 'j2elem']):
                tanh = np.tanh(1.0 - R/Rc)
                dgdr = - 3.0 / Rc * ((1.0-tanh**2) * tanh**2)[:, None] * dRdr
                dgdr = np.array([np.sum(dgdr_j, axis=0)
                                 for dgdr_j in np.split(dgdr, neigh2j[1:])])
                yield block_diag(*np.split(dgdr, j2elem[1:])).reshape(
                    n_atom, -1, 3).transpose(1, 0, 2)

        if self._order == 0:
            return symmetry_function_type1,
        elif self._order == 1:
            return symmetry_function_type1, diff_symmetry_function_type1

    def _make_symmetry_function_type2(self, Rc, eta, Rs):
        def symmetry_function_type2(structure):
            for R, neigh2elem in structure.get_neighbor_info(
                    Rc, ['distance', 'neigh2elem']):
                g = np.exp(-eta*(R-Rs)**2) * np.tanh(1.0 - R/Rc)**3
                yield [np.sum(g_e) for g_e in np.split(g, neigh2elem[1:])]

        def diff_symmetry_function_type2(structure):
            n_atom = len(structure)
            for R, dRdr, neigh2j, j2elem in structure.get_neighbor_info(
                    Rc, ['distance', 'diff_distance', 'neigh2j', 'j2elem']):
                tanh = np.tanh(1.0 - R / Rc)
                dgdr = (np.exp(-eta*(R-Rs)**2)
                        * tanh**2
                        * (-2.0*eta*(R-Rs)*tanh + 3.0/Rc*(tanh**2-1.0))
                        )[:, None] * dRdr
                dgdr = np.array([np.sum(dgdr_j, axis=0)
                                 for dgdr_j in np.split(dgdr, neigh2j[1:])])
                yield block_diag(*np.split(dgdr, j2elem[1:])).reshape(
                    n_atom, -1, 3).transpose(1, 0, 2)

        if self._order == 0:
            return symmetry_function_type2,
        elif self._order == 1:
            return symmetry_function_type2, diff_symmetry_function_type2

    def _make_symmetry_function_type4(self, Rc, eta, lambda_, zeta):
        def symmetry_function_type4(structure):
            for R, cos, neigh2elem in structure.get_neighbor_info(
                    Rc, ['distance', 'cosine', 'neigh2elem']):
                ang = np.triu(1.0 + lambda_*cos, k=1)
                rad = np.exp(-eta*R**2) * np.tanh(1.0 - R/Rc)**3
                g = 2.0**(1-zeta) * ang**zeta * rad[:, None] * rad[None, :]

                yield [np.sum(g_ee)
                       for j, g_e in enumerate(np.split(g, neigh2elem[1:]))
                       for k, g_ee in enumerate(np.split(g_e, neigh2elem[1:],
                                                         axis=1))
                       if j <= k]

        def diff_symmetry_function_type4(structure):
            n_elem = len(structure.elements)
            n_comb = n_elem * (n_elem+1) // 2
            n_atom = len(structure)
            for R, dRdr, cos, dcosdr, neigh2elem, neigh2j, j2elem in (
                    structure.get_neighbor_info(Rc, [
                        'distance', 'diff_distance', 'cosine', 'diff_cosine',
                        'neigh2elem', 'neigh2j', 'j2elem'])):
                tanh = np.tanh(1.0 - R / Rc)
                ang = 1.0 + lambda_*cos
                np.fill_diagonal(ang, 0.0)
                rad1 = np.exp(-eta*R**2) * tanh**3
                rad2 = (np.exp(-eta*R**2)
                        * tanh**2
                        * (-2.0*eta*R*tanh + 3.0/Rc*(tanh**2-1.0)))

                dgdr_radial_part = (2.0**(1-zeta)
                                    * ang[:, :, None]**zeta
                                    * rad2[:, None, None]
                                    * rad1[None, :, None]
                                    * dRdr[:, None, :])
                dgdr_angular_part = (zeta * lambda_
                                     * 2.0**(1-zeta)
                                     * ang[:, :, None]**(zeta-1)
                                     * rad1[:, None, None]
                                     * rad1[None, :, None]
                                     * dcosdr)
                dgdr = dgdr_radial_part + dgdr_angular_part
                dgdr = np.array(
                    [[np.sum(dgdr_je, axis=(0, 1))
                      for dgdr_je in np.split(dgdr_j, neigh2elem[1:], axis=1)]
                     for dgdr_j in np.split(dgdr, neigh2j[1:])])

                dGdr = np.zeros((n_comb, n_atom, 3))
                j2elem = np.append(j2elem, 0)
                for c, (j, k) in enumerate(combinations_with_replacement(
                        range(n_elem), 2)):
                    dGdr[c, j2elem[j]:j2elem[j+1]] = (
                        dgdr[j2elem[j]:j2elem[j+1], k])
                    dGdr[c, j2elem[k]:j2elem[k+1]] = (
                        dgdr[j2elem[k]:j2elem[k+1], j])
                yield dGdr

        if self._order == 0:
            return symmetry_function_type4,
        elif self._order == 1:
            return symmetry_function_type4, diff_symmetry_function_type4

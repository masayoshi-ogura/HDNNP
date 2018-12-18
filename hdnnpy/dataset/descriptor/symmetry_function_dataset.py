# coding: utf-8

from itertools import (chain,
                       combinations_with_replacement,
                       )

import numpy as np
from scipy.linalg import block_diag

from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)
from hdnnpy.utils import (MPI,
                          pprint,
                          )


class SymmetryFunctionDataset(DescriptorDatasetBase):
    DESCRIPTORS = ['sym_func', 'derivative']
    name = 'symmetry_function'

    def __init__(self, order=0):
        assert 0 <= order <= 1
        super().__init__(order)
        self._descriptors = self.DESCRIPTORS[: order+1]
        self._func_param_map = {}

    @property
    def function_names(self):
        return list(self._func_param_map.keys())

    @property
    def params(self):
        return self._func_param_map

    def clear(self):
        super().clear()
        self._func_param_map.clear()

    def dump_params(self):
        # todo
        pass

    def generate_feature_keys(self, elements):
        feature_keys = []
        for function_name, params_set in self._func_param_map.items():
            for params in params_set:
                param_key = '/'.join(map(str, params))
                if function_name in ['type1', 'type2']:
                    for element in elements:
                        key = ':'.join([function_name, param_key, element])
                        feature_keys.append(key)

                elif function_name in ['type4']:
                    for elements in combinations_with_replacement(elements, 2):
                        element_key = '/'.join(elements)
                        key = ':'.join(
                            [function_name, param_key, element_key])
                        feature_keys.append(key)
        return feature_keys

    def load(self, file_path, verbose=True):
        if MPI.rank == 0:
            ndarray = np.load(file_path)
            for i in range(self._order + 1):
                self._dataset.append(ndarray[self._descriptors[i]])
            self._elemental_composition = list(
                ndarray['elemental_composition'])
            self._elements = list(ndarray['elements'])
            self._feature_keys = list(ndarray['feature_keys'])
            self._func_param_map = ndarray['func_param_map'].item()
            self._tag = ndarray['tag'].item()
            if verbose:
                pprint(f'Loaded symmetry function dataset from {file_path}.')

    def make(self, structures, verbose=True, **func_param_map):
        assert func_param_map
        self._elemental_composition = structures[0].get_chemical_symbols()
        self._tag = structures[0].info['tag']
        self._elements = sorted(set(self._elemental_composition))
        self._func_param_map = func_param_map.copy()
        self._feature_keys = self.generate_feature_keys(self._elements)

        n_sample = len(structures)
        n_atom = len(structures[0])
        count = np.array([(n_sample + i) // MPI.size
                          for i in range(MPI.size)[::-1]], dtype=np.int32)
        s = count[: MPI.rank].sum()
        e = count[: MPI.rank+1].sum()
        structures = structures[s:e]

        for i, send_data in enumerate(self._calculate_descriptors(structures)):
            recv_data = None
            if MPI.rank == 0:
                shape = (n_sample, n_atom, self.n_feature, *(n_atom, 3) * i)
                data = np.empty(shape, dtype=np.float32)
                recv_data = (data, count * data[0].size)
                MPI.comm.Gatherv(send_data, recv_data, root=0)
                self._dataset.append(data)
            else:
                MPI.comm.Gatherv(send_data, recv_data, root=0)

        if verbose:
            pprint('Calculated symmetry function dataset.')

    def save(self, file_path, verbose=True):
        if MPI.rank == 0:
            if not self.has_data:
                raise RuntimeError('This dataset does not have any data.')

            data = {property_: data for property_, data
                    in zip(self._descriptors, self._dataset)}
            info = {
                'elemental_composition': self._elemental_composition,
                'elements': self._elements,
                'feature_keys': self._feature_keys,
                'func_param_map': self._func_param_map,
                'tag': self._tag,
                }
            np.savez(file_path, **data, **info)
            if verbose:
                pprint(f'Saved symmetry function dataset to {file_path}.')

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
            for i, structure in enumerate(structures):
                for j, g in enumerate(zip(*[func[0](structure)
                                            for func in functions])):
                    G[i, j] = list(chain.from_iterable(g))
            yield G

        if self._order >= 1:
            dGdr = np.zeros((n_sample, n_atom, self.n_feature, n_atom, 3),
                            dtype=np.float32)
            for i, structure in enumerate(structures):
                for j, dgdr in enumerate(zip(*[func[1](structure)
                                               for func in functions])):
                    dGdr[i, j] = list(chain.from_iterable(dgdr))
            yield dGdr

        for structure in structures:
            structure.clear_cache()

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

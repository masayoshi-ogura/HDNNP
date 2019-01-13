# coding: utf-8

"""Symmetry function dataset for descriptor of HDNNP."""

from itertools import combinations_with_replacement

import chainer
import chainer.functions as F
import numpy as np
from tqdm import tqdm

from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)
from hdnnpy.utils import (MPI, pprint, recv_chunk, send_chunk)


class SymmetryFunctionDataset(DescriptorDatasetBase):
    """Symmetry function dataset for descriptor of HDNNP."""
    DESCRIPTORS = ['sym_func', 'derivative', 'second_derivative']
    """list [str]: Names of descriptors for each derivative order."""
    name = 'symmetry_function'
    """str: Name of this descriptor class."""

    def __init__(self, order, structures, **func_param_map):
        """
        It accepts 0 or 2 for ``order``.

        | Each symmetry function requires following parameters.
        | Pass parameters you want to use for the dataset as keyword
          arguments ``func_param_map``.

        * type1: :math:`R_c`
        * type2: :math:`R_c, \eta, R_s`
        * type4: :math:`R_c, \eta, \lambda, \zeta`

        Args:
            order (int): passed to super class.
            structures (list [AtomicStructure]): passed to super class.
            **func_param_map (list [tuple]):
                parameter sets for each type of symmetry function.

        References:
            Symmetry function was proposed by Behler *et al.* in
            `this paper`_ as a descriptor of HDNNP. Please see here for
            details of each symmetry function.

        .. _`this paper`:
            https://onlinelibrary.wiley.com/doi/full/10.1002/qua.24890
        """
        assert 0 <= order <= 2
        assert func_param_map
        super().__init__(order, structures)
        self._func_param_map = func_param_map.copy()
        self._feature_keys = self.generate_feature_keys(self._elements)

    @property
    def function_names(self):
        """list [str]: Names of symmetry functions this instance
        calculates or has calculated."""
        return list(self._func_param_map.keys())

    @property
    def params(self):
        """dict [list [tuple]]]: Mapping from symmetry function name to
        its parameters."""
        return self._func_param_map

    def generate_feature_keys(self, elements):
        """Generate feature keys from given elements and parameters.

        | parameters given at initialization are used.
        | This method is used to initialize instance and expand feature
          dimension in
          :class:`~hdnnpy.dataset.hdnnp_dataset.HDNNPDataset`.

        Args:
            elements (list [str]): Unique list of elements. It should be
                sorted alphabetically.

        Returns:
            list [str]: Generated feature keys in a format
            like ``<func_name>:<parameters>:<elements>``.
        """
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
        """Calculate & retain symmetry functions.

        | It calculates symmetry functions by data-parallel using MPI
          communication.
        | The calculated dataset is retained in only root MPI process.

        Args:
            verbose (bool, optional): Print log to stdout.
        """
        dataset = []
        for structure in tqdm(self._structures,
                              ascii=True, desc=f'Process #{MPI.rank}',
                              leave=False, position=MPI.rank):
            dataset.append(self._calculate_descriptors(structure))

        for i, data_list in enumerate(zip(*dataset)):
            n_atom = len(self._structures[0])
            shape = (n_atom, self.n_feature, *(3*n_atom,) * i)
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

    def _calculate_descriptors(self, structure):
        """Main method of calculating symmetry functions."""
        generators = []
        for name, params_set in self._func_param_map.items():
            for params in params_set:
                generators.append(eval(
                    f'self._symmetry_function_{name}')(structure, *params))

        dataset = [np.concatenate([next(gen).data
                                   for gen in generators]).swapaxes(0, 1)
                   for _ in range(self._order + 1)]

        structure.clear_cache()

        return dataset

    def differentiate(func):
        def wrapper(self, structure, Rc, *params):
            G = func(self, structure, Rc, *params)
            yield F.stack([F.stack(g) for g in G])

            n_atom = len(G[0])
            r = []
            neigh2j = []
            for r_, neigh2j_ in structure.get_neighbor_info(
                    Rc, ['distance_vector', 'neigh2j']):
                r.append(r_)
                neigh2j.append(neigh2j_)

            dG = []
            for g in G:
                grad = chainer.grad(
                    g, r, enable_double_backprop=self._order >= 2)
                dg = [F.concat([F.sum(dg_, axis=0) for dg_
                                in F.split_axis(grad_, neigh2j_[1:],
                                                axis=0)],
                               axis=0)
                      for grad_, neigh2j_ in zip(grad, neigh2j)]
                dG.append(dg)
            yield F.stack([F.stack(dg) for dg in dG])

            d2G = []
            for dg in dG:
                d2g = []
                for i in range(3 * n_atom):
                    grad = chainer.grad(
                        [dg_[i] for dg_ in dg], r,
                        enable_double_backprop=self._order >= 3)
                    d2g_ = [F.concat([F.sum(d2g_, axis=0) for d2g_
                                      in F.split_axis(grad_, neigh2j_[1:],
                                                      axis=0)],
                                     axis=0)
                            for grad_, neigh2j_ in zip(grad, neigh2j)]
                    d2g.append(d2g_)
                d2G.append(d2g)
            yield F.stack([F.stack([F.stack(d2g_) for d2g_ in d2g])
                           for d2g in d2G]).transpose(0, 2, 1, 3)

        return wrapper

    @differentiate
    def _symmetry_function_type1(self, structure, Rc):
        """Symmetry function type1 for specific parameters."""
        G = []
        for R, neigh2elem in structure.get_neighbor_info(
                Rc, ['distance', 'neigh2elem']):
            g = F.tanh(1.0 - R/Rc) ** 3
            g = [F.sum(g_) for g_ in F.split_axis(g, neigh2elem[1:], axis=0)]
            G.append(g)
        return list(zip(*G))

    @differentiate
    def _symmetry_function_type2(self, structure, Rc, eta, Rs):
        """Symmetry function type2 for specific parameters."""
        G = []
        for R, neigh2elem in structure.get_neighbor_info(
                Rc, ['distance', 'neigh2elem']):
            g = F.exp(-eta*(R-Rs)**2) * F.tanh(1.0 - R/Rc)**3
            g = [F.sum(g_) for g_ in F.split_axis(g, neigh2elem[1:], axis=0)]
            G.append(g)
        return list(zip(*G))

    @differentiate
    def _symmetry_function_type4(self, structure, Rc, eta, lambda_, zeta):
        """Symmetry function type4 for specific parameters."""
        G = []
        for r, R, neigh2elem in structure.get_neighbor_info(
                Rc, ['distance_vector', 'distance', 'neigh2elem']):
            cos = (r/F.expand_dims(R, axis=1)) @ (r.T/R)
            if zeta == 1:
                ang = (1.0 + lambda_*cos)
            else:
                ang = (1.0 + lambda_*cos) ** zeta
            g = (2.0 ** (1-zeta)
                 * ang
                 * F.expand_dims(F.exp(-eta*R**2) * F.tanh(1.0 - R/Rc)**3,
                                 axis=1)
                 * F.expand_dims(F.exp(-eta*R**2) * F.tanh(1.0 - R/Rc)**3,
                                 axis=0))
            triu = np.triu(np.ones_like(cos.data), k=1)
            g = F.where(triu.astype(np.bool), g, triu)
            g = [F.sum(g__)
                 for j, g_
                 in enumerate(F.split_axis(g, neigh2elem[1:], axis=0))
                 for k, g__
                 in enumerate(F.split_axis(g_, neigh2elem[1:], axis=1))
                 if j <= k]
            G.append(g)
        return list(zip(*G))

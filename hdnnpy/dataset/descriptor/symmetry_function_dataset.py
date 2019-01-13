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
        n_sample = len(self._structures)
        n_atom = len(self._structures[0])
        slices = [slice(i[0], i[-1]+1)
                  for i in np.array_split(range(n_sample), MPI.size)]
        structures = self._structures[slices[MPI.rank]]

        for i, send_data in enumerate(
                self._calculate_descriptors(structures, verbose)):
            shape = (n_atom, self.n_feature, *(n_atom*3,) * i)
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

    def _calculate_descriptors(self, structures, verbose):
        """Main method of calculating symmetry functions."""
        generators = []
        for structure in structures:
            nst = []
            for name, params_set in self._func_param_map.items():
                for params in params_set:
                    nst.append(eval(
                        f'self._symmetry_function_{name}')(structure, *params))
            generators.append(nst)

        if self._order >= 0:
            if verbose:
                pprint('Calculate symmetry function: 0th order')
            G = []
            for gen_list in tqdm(generators,
                                 ascii=True, desc=f'Process #{MPI.rank}',
                                 leave=False, position=MPI.rank):
                G.append(np.concatenate(
                    [next(gen).data for gen in gen_list], axis=1))
            yield np.stack(G)

        if self._order >= 1:
            if verbose:
                pprint('Calculate symmetry function: 1st order')
            dG = []
            for gen_list in tqdm(generators,
                                 ascii=True, desc=f'Process #{MPI.rank}',
                                 leave=False, position=MPI.rank):
                dG.append(np.concatenate(
                    [next(gen).data for gen in gen_list], axis=1))
            yield np.stack(dG)

        if self._order >= 2:
            if verbose:
                pprint('Calculate symmetry function: 2nd order')
            d2G = []
            for gen_list in tqdm(generators,
                                 ascii=True, desc=f'Process #{MPI.rank}',
                                 leave=False, position=MPI.rank):
                d2G.append(np.concatenate(
                    [next(gen).data for gen in gen_list], axis=1))
            yield np.stack(d2G)

        for structure in structures:
            structure.clear_cache()

    def differentiate(func):
        def wrapper(self, structure, Rc, *params):
            G = func(self, structure, Rc, *params)
            yield F.stack(G)

            dG = []
            for g, (r, neigh2j) in zip(G, structure.get_neighbor_info(
                    Rc, ['distance_vector', 'neigh2j'])):
                dg = []
                for g_ in g:
                    grad, = chainer.grad(
                        [g_], [r],
                        enable_double_backprop=self._order >= 2)
                    dg.append(F.concat([
                        F.sum(dg_, axis=0) for dg_
                        in F.split_axis(grad, neigh2j[1:], axis=0)],
                        axis=0))
                dG.append(F.stack(dg))
            yield F.stack(dG)

            d2G = []
            for dg, (r, neigh2j) in zip(dG, structure.get_neighbor_info(
                    Rc, ['distance_vector', 'neigh2j'])):
                d2g = []
                for dg_ in dg:
                    d2g_ = []
                    for dg__ in dg_:
                        grad, = chainer.grad(
                            [dg__], [r],
                            enable_double_backprop=self._order >= 3)
                        d2g_.append(F.concat([
                            F.sum(d2g_, axis=0) for d2g_
                            in F.split_axis(grad, neigh2j[1:], axis=0)],
                            axis=0))
                    d2g.append(F.stack(d2g_))
                d2G.append(F.stack(d2g))
            yield F.stack(d2G)

        return wrapper

    @differentiate
    def _symmetry_function_type1(self, structure, Rc):
        """Symmetry function type1 for specific parameters."""
        G = []
        for R, neigh2elem in structure.get_neighbor_info(
                Rc, ['distance', 'neigh2elem']):
            g = F.tanh(1.0 - R/Rc) ** 3
            g = [F.sum(g_) for g_ in F.split_axis(g, neigh2elem[1:], axis=0)]
            G.append(F.stack(g))
        return G

    @differentiate
    def _symmetry_function_type2(self, structure, Rc, eta, Rs):
        """Symmetry function type2 for specific parameters."""
        G = []
        for R, neigh2elem in structure.get_neighbor_info(
                Rc, ['distance', 'neigh2elem']):
            g = F.exp(-eta*(R-Rs)**2) * F.tanh(1.0 - R/Rc)**3
            g = [F.sum(g_) for g_ in F.split_axis(g, neigh2elem[1:], axis=0)]
            G.append(F.stack(g))
        return G

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
            G.append(F.stack(g))
        return G

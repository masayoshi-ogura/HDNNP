# coding: utf-8

"""Weighted symmetry function dataset for descriptor of HDNNP."""

import chainer
import chainer.functions as F
import numpy as np

from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)


class WeightedSymmetryFunctionDataset(DescriptorDatasetBase):
    """Weighted symmetry function dataset for descriptor of HDNNP."""
    DESCRIPTORS = ['sym_func', 'derivative', 'second_derivative']
    """list [str]: Names of descriptors for each derivative order."""
    name = 'weighted_symmetry_function'
    """str: Name of this descriptor class."""

    def __init__(self, order, structures, **func_param_map):
        """
        It accepts 0 or 2 for ``order``.

        | Each weighted symmetry function requires following parameters.
        | Pass parameters you want to use for the dataset as keyword
          arguments ``func_param_map``.

        * type1: :math:`R_c`
        * type2: :math:`R_c, \eta, R_s`
        * type4: :math:`R_c, \eta, \lambda, \zeta`

        Args:
            order (int): passed to super class.
            structures (list [AtomicStructure]): passed to super class.
            **func_param_map (list [tuple]):
                parameter sets for each type of weighted symmetry function.

        References:
            Weighted symmetry function was proposed by Gastegger *et al.* in
            `this paper`_ as a descriptor of HDNNP. Please see here for
            details of weighted symmetry function.

        .. _`this paper`:
            https://doi.org/10.1063/1.5019667
        """
        assert 0 <= order <= 2
        assert func_param_map
        super().__init__(order, structures)
        self._func_param_map = func_param_map.copy()
        self._feature_keys = self.generate_feature_keys(self._elements)

    @property
    def function_names(self):
        """list [str]: Names of weighted symmetry functions this
        instance calculates or has calculated."""
        return list(self._func_param_map.keys())

    @property
    def params(self):
        """dict [list [tuple]]]: Mapping from weighted symmetry function
        name to its parameters."""
        return self._func_param_map

    def calculate_descriptors(self, structure):
        """Calculate required descriptors for a structure data.

        Args:
            structure (AtomicStructure):
                A structure data to calculate descriptors.

        Returns:
            list [~numpy.ndarray]: Calculated descriptors.
            The length is the same as ``order`` given at initialization.
        """
        generators = []
        for name, params_set in self._func_param_map.items():
            for params in params_set:
                generators.append(eval(
                    f'self._weighted_symmetry_function_{name}'
                    )(structure, *params))

        dataset = [np.stack([next(gen).data
                             for gen in generators]).swapaxes(0, 1)
                   for _ in range(self._order + 1)]

        structure.clear_cache()

        return dataset

    def generate_feature_keys(self, _):
        """Generate feature keys from given elements and parameters.

        | parameters given at initialization are used.
        | This method is used to initialize instance and expand feature
          dimension in
          :class:`~hdnnpy.dataset.hdnnp_dataset.HDNNPDataset`.

        Returns:
            list [str]: Generated feature keys in a format
            like ``<func_name>:<parameters>``.
        """
        feature_keys = []
        for function_name, params_set in self._func_param_map.items():
            for params in params_set:
                param_key = '/'.join(map(str, params))
                key = ':'.join([function_name, param_key])
                feature_keys.append(key)
        return feature_keys

    def differentiate(func):
        """Decorator function to differentiate weighted symmetry
        function."""
        def wrapper(self, structure, Rc, *params):
            differentiate_more = self._order > 0
            with chainer.using_config('enable_backprop', differentiate_more):
                G = func(self, structure, Rc, *params)
                yield F.stack(G)

            n_atom = len(G)
            r = []
            j_indices = []
            for r_, j_idx in structure.get_neighbor_info(
                    Rc, ['distance_vector', 'j_indices']):
                r.append(r_)
                j_indices.append(j_idx)

            differentiate_more = self._order > 1
            with chainer.using_config('enable_backprop', differentiate_more):
                with chainer.force_backprop_mode():
                    grad = chainer.grad(
                        G, r, enable_double_backprop=differentiate_more)
                dG = [F.concat([F.sum(dg_, axis=0) for dg_
                                in F.split_axis(grad_, j_idx[1:], axis=0)],
                               axis=0)
                      for grad_, j_idx in zip(grad, j_indices)]
                yield F.stack(dG)

            differentiate_more = self._order > 2
            with chainer.using_config('enable_backprop', differentiate_more):
                d2G = []
                for i in range(3 * n_atom):
                    with chainer.force_backprop_mode():
                        grad = chainer.grad(
                            [dg[i] for dg in dG], r,
                            enable_double_backprop=differentiate_more)
                    d2g = [F.concat([F.sum(d2g_, axis=0) for d2g_
                                     in F.split_axis(grad_, j_idx[1:],
                                                     axis=0)],
                                    axis=0)
                           for grad_, j_idx in zip(grad, j_indices)]
                    d2G.append(d2g)
                yield F.stack([F.stack(d2g) for d2g in d2G]).transpose(1, 0, 2)

        return wrapper

    @differentiate
    def _weighted_symmetry_function_type1(self, structure, Rc):
        """Weighted symmetry function type1 for specific parameters."""
        G = []
        for z, fc in structure.get_neighbor_info(
                Rc, ['atomic_number', 'cutoff_function']):
            g = z * fc
            G.append(F.sum(g))
        return G

    @differentiate
    def _weighted_symmetry_function_type2(self, structure, Rc, eta, Rs):
        """Weighted symmetry function type2 for specific parameters."""
        G = []
        for z, R, fc in structure.get_neighbor_info(
                Rc, ['atomic_number', 'distance', 'cutoff_function']):
            g = z * F.exp(-eta*(R-Rs)**2) * fc
            G.append(F.sum(g))
        return G

    @differentiate
    def _weighted_symmetry_function_type4(
            self, structure, Rc, eta, lambda_, zeta):
        """Weighted symmetry function type4 for specific parameters."""
        G = []
        for z, r, R, fc in structure.get_neighbor_info(
                Rc, ['atomic_number', 'distance_vector', 'distance',
                     'cutoff_function']):
            cos = (r/F.expand_dims(R, axis=1)) @ (r.T/R)
            if zeta == 1:
                ang = (1.0 + lambda_*cos)
            else:
                ang = (1.0 + lambda_*cos) ** zeta
            g = (2.0 ** (1-zeta)
                 * z[:, None] * z[None, :]
                 * ang
                 * F.expand_dims(F.exp(-eta*R**2) * fc, axis=1)
                 * F.expand_dims(F.exp(-eta*R**2) * fc, axis=0))
            triu = np.triu(np.ones_like(cos.data), k=1)
            g = F.where(triu.astype(np.bool), g, triu)
            G.append(F.sum(g))
        return G

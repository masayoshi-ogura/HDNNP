# coding: utf-8

"""Symmetry function dataset for descriptor of HDNNP."""

from itertools import combinations_with_replacement

import chainer
import chainer.functions as F
import numpy as np

from hdnnpy.dataset.descriptor.descriptor_dataset_base import (
    DescriptorDatasetBase)


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
                    f'self._symmetry_function_{name}')(structure, *params))

        dataset = [np.concatenate([next(gen).data
                                   for gen in generators]).swapaxes(0, 1)
                   for _ in range(self._order + 1)]

        structure.clear_cache()

        return dataset

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

    def differentiate(func):
        """Decorator function to differentiate symmetry function."""
        def wrapper(self, structure, Rc, *params):
            differentiate_more = self._order > 0
            with chainer.using_config('enable_backprop', differentiate_more):
                G = func(self, structure, Rc, *params)
                yield F.stack([F.stack(g) for g in G])

            n_atom = len(G[0])
            diff_positions = []
            diff_indices = []
            for i_pos, i_idx, j_pos, j_idx in structure.get_neighbor_info(
                    Rc,
                    ['i_positions', 'i_indices', 'j_positions', 'j_indices']
                ):
                diff_positions.extend([i_pos, j_pos])
                diff_indices.extend([i_idx, j_idx])

            differentiate_more = self._order > 1
            with chainer.using_config('enable_backprop', differentiate_more):
                dG = []
                for g in G:
                    with chainer.force_backprop_mode():
                        grad = chainer.grad(
                            g, diff_positions,
                            enable_double_backprop=differentiate_more)
                    dg = [
                        # by center atom itself
                        F.concat([
                            F.sum(dg_, axis=0)
                            for dg_ in F.split_axis(
                                grad[2*i], diff_indices[2*i][1:], axis=0
                                )
                            ], axis=0)
                        # by neighbor atoms
                        + F.concat([
                            F.sum(dg_, axis=0)
                            for dg_ in F.split_axis(
                                grad[2*i+1], diff_indices[2*i+1][1:], axis=0
                                )
                            ], axis=0)
                        for i in range(n_atom)
                    ]
                    dG.append(dg)
                yield F.stack([F.stack(dg) for dg in dG])

            differentiate_more = self._order > 2
            with chainer.using_config('enable_backprop', differentiate_more):
                d2G = []
                for dg in dG:
                    d2g = []
                    for j in range(3 * n_atom):
                        with chainer.force_backprop_mode():
                            grad = chainer.grad(
                                [dg_[j] for dg_ in dg], diff_positions,
                                enable_double_backprop=differentiate_more)
                        d2g_ = [
                            # by center atom itself
                            F.concat([
                                F.sum(d2g_, axis=0)
                                for d2g_ in F.split_axis(
                                    grad[2*i], diff_indices[2*i][1:], axis=0
                                    )
                                ], axis=0)
                            # by neighbor atoms
                            + F.concat([
                                F.sum(d2g_, axis=0)
                                for d2g_ in F.split_axis(
                                    grad[2*i+1], diff_indices[2*i+1][1:], axis=0
                                    )
                                ], axis=0)
                            for i in range(n_atom)
                        ]
                        d2g.append(d2g_)
                    d2G.append(d2g)
                yield F.stack([F.stack([F.stack(d2g_) for d2g_ in d2g])
                               for d2g in d2G]).transpose(0, 2, 1, 3)

        return wrapper

    @differentiate
    def _symmetry_function_type1(self, structure, Rc):
        """Symmetry function type1 for specific parameters."""
        G = []
        for fc, element_indices in structure.get_neighbor_info(
                Rc, ['cutoff_function', 'element_indices']):
            g = fc
            g = [F.sum(g_) for g_
                 in F.split_axis(g, element_indices[1:], axis=0)]
            G.append(g)
        return list(zip(*G))

    @differentiate
    def _symmetry_function_type2(self, structure, Rc, eta, Rs):
        """Symmetry function type2 for specific parameters."""
        G = []
        for R, fc, element_indices in structure.get_neighbor_info(
                Rc, ['distance', 'cutoff_function', 'element_indices']):
            g = F.exp(-eta*(R-Rs)**2) * fc
            g = [F.sum(g_) for g_
                 in F.split_axis(g, element_indices[1:], axis=0)]
            G.append(g)
        return list(zip(*G))

    @differentiate
    def _symmetry_function_type4(self, structure, Rc, eta, lambda_, zeta):
        """Symmetry function type4 for specific parameters."""
        G = []
        for r, R, fc, element_indices in structure.get_neighbor_info(
                Rc, ['distance_vector', 'distance', 'cutoff_function',
                     'element_indices']):
            cos = (r/F.expand_dims(R, axis=1)) @ (r.T/R)
            if zeta == 1:
                ang = (1.0 + lambda_*cos)
            else:
                ang = (1.0 + lambda_*cos) ** zeta
            g = (2.0 ** (1-zeta)
                 * ang
                 * F.expand_dims(F.exp(-eta*R**2) * fc, axis=1)
                 * F.expand_dims(F.exp(-eta*R**2) * fc, axis=0))
            mask = 1 - np.eye(cos.data.shape[0], dtype=np.float32)
            g = F.where(mask.astype(np.bool), g, mask)
            g = [F.sum(g__)
                 for j, g_
                 in enumerate(F.split_axis(g, element_indices[1:], axis=0))
                 for k, g__
                 in enumerate(F.split_axis(g_, element_indices[1:], axis=1))
                 if j <= k]
            G.append(g)
        return list(zip(*G))

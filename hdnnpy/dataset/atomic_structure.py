# coding: utf-8

"""Wrapper class of ase.Atoms."""

from ase.calculators.singlepoint import SinglePointCalculator
import ase.io
import ase.neighborlist
import chainer
import chainer.functions as F
import numpy as np


class AtomicStructure(object):
    """Wrapper class of ase.Atoms."""
    def __init__(self, atoms):
        """
        | It wraps :obj:`ase.Atoms` object to define additional methods
          and attributes.
        | Before wrapping, it sorts atoms by element alphabetically.
        | It stores calculated neighbor information such as distance,
          indices.

        Args:
            atoms (~ase.Atoms): an object to wrap.
        """
        tags = atoms.get_chemical_symbols()
        deco = sorted([(tag, i) for i, tag in enumerate(tags)])
        indices = [i for tag, i in deco]
        self._atoms = atoms[indices]

        results = {}
        calculator = atoms.get_calculator()
        if calculator:
            for key, value in calculator.results.items():
                if key in atoms.arrays:
                    results[key] = value[indices]
                else:
                    results[key] = value
        self._atoms.set_calculator(
            SinglePointCalculator(self._atoms, **results))

        self._cache = {}

    def __getattr__(self, item):
        return getattr(self._atoms, item)

    def __getstate__(self):
        return self._atoms

    def __len__(self):
        return len(self._atoms)

    def __setstate__(self, state):
        self._atoms = state
        self._cache = {}

    @property
    def elements(self):
        """list [str]: Elements included in a cell."""
        return sorted(set(self._atoms.get_chemical_symbols()))

    def clear_cache(self, cutoff_distance=None):
        """Clear up cached neighbor information in this instance.

        Args:
            cutoff_distance (float, optional):
                It clears the corresponding cached data if specified,
                otherwise it clears all cached data.
        """
        if cutoff_distance:
            self._cache[cutoff_distance].clear()
        else:
            self._cache.clear()

    def get_neighbor_info(self, cutoff_distance, geometry_keys):
        """Calculate or return cached data.

        | If there is no cached data, calculate it as necessary.
        | The calculated result is cached, and retained unless
          you use :meth:`clear_cache` method.

        Args:
            cutoff_distance (float):
                It calculates the geometry for the neighboring atoms
                within this value of each atom in a cell.
            geometry_keys (list [str]):
                A list of atomic geometries to calculate between an atom
                and its neighboring atoms.

        Returns:
            Iterator [tuple]: Neighbor information required by
            ``geometry_keys`` for each atom in a cell.
        """
        ret = []
        for key in geometry_keys:
            if (cutoff_distance not in self._cache
                    or key not in self._cache[cutoff_distance]):
                self._calculate_neighbors(cutoff_distance)
            ret.append(self._cache[cutoff_distance][key])
        for neighbor_info in zip(*ret):
            yield neighbor_info

    @classmethod
    def read_xyz(cls, file_path):
        """Read .xyz format file and make a list of instances.

        Parses .xyz format file using :func:`ase.io.iread` and wraps it
        by this class.

        Args:
            file_path (~pathlib.Path):
                File path to read atomic structures.

        Returns:
            list [AtomicStructure]: Initialized instances.
        """
        return [cls(atoms) for atoms
                in ase.io.iread(str(file_path), index=':', format='xyz')]

    def _calculate_neighbors(self, cutoff_distance):
        """Calculate distance to one neighboring atom and store indices
        of neighboring atoms."""
        symbols = self._atoms.get_chemical_symbols()
        elements = sorted(set(symbols))
        atomic_numbers = self._atoms.get_atomic_numbers()
        index_element_map = [elements.index(element) for element in symbols]

        i_list, j_list, S_list = ase.neighborlist.neighbor_list(
            'ijS', self._atoms, cutoff_distance)

        sort_indices = np.lexsort((j_list, i_list))  # sort by i, then by j
        i_list = i_list[sort_indices]
        j_list = j_list[sort_indices]
        S_list = S_list[sort_indices]
        elem_list = np.array([index_element_map[idx] for idx in j_list])

        i_list, i_indices = np.unique(i_list, return_index=True)
        j_list = np.split(j_list, i_indices[1:])
        S_list = np.split(S_list, i_indices[1:])
        r_i_list = [
            chainer.Variable(np.array([
                self._atoms.positions[i]
                ], dtype=np.float32))
            for i in i_list
            ]
        r_j_list = [
            chainer.Variable(np.array([
                self._atoms.positions[j] + np.dot(s, self._atoms.cell)
                for j, s in zip(j_indices, shift)
                ], dtype=np.float32))
            for j_indices, shift in zip(j_list, S_list)
            ]
        distance_vector = [r_j - r_i for r_i, r_j in zip(r_i_list, r_j_list)]
        distance = [F.sqrt(F.sum(r**2, axis=1)) for r in distance_vector]
        cutoff_function = [F.tanh(1.0 - R/cutoff_distance)**3
                           for R in distance]
        elem_list = np.split(elem_list, i_indices[1:])

        self._cache[cutoff_distance] = {
            'distance_vector': distance_vector,
            'distance': distance,
            'cutoff_function': cutoff_function,
            'element_indices': [np.searchsorted(elem, range(len(elements)))
                                for elem in elem_list],
            'i_positions': r_i_list,
            'i_indices': i_list,
            'i_indices': [np.searchsorted([i], range(len(symbols)))
                          for i in i_list],
            'j_positions': r_j_list,
            'j_indices': [np.searchsorted(j, range(len(symbols)))
                          for j in j_list],
            'atomic_number': [
                np.apply_along_axis(lambda x: atomic_numbers[x], 0, j)
                for j in j_list],
            }

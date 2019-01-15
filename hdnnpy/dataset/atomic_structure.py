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
                if key in ['distance_vector', 'distance', 'neigh2elem',
                           'neigh2j']:
                    self._calculate_distance(cutoff_distance)
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

    def _calculate_distance(self, cutoff_distance):
        """Calculate distance to one neighboring atom and store indices
        of neighboring atoms."""
        symbols = self._atoms.get_chemical_symbols()
        elements = sorted(set(symbols))
        index_element_map = [elements.index(element) for element in symbols]

        i_list, j_list, D_list = ase.neighborlist.neighbor_list(
            'ijD', self._atoms, cutoff_distance)

        sort_indices = np.lexsort((j_list, i_list))
        i_list = i_list[sort_indices]
        j_list = j_list[sort_indices]
        D_list = D_list[sort_indices]
        elem_list = np.array([index_element_map[idx] for idx in j_list])

        i_indices = np.unique(i_list, return_index=True)[1]
        j_list = np.split(j_list, i_indices[1:])
        distance_vector = [chainer.Variable(r.astype(np.float32))
                           for r in np.split(D_list, i_indices[1:])]
        distance = [F.sqrt(F.sum(r**2, axis=1)) for r in distance_vector]
        elem_list = np.split(elem_list, i_indices[1:])

        neigh2j = []
        neigh2elem = []
        for j, elem in zip(j_list, elem_list):
            neigh2j.append(np.searchsorted(j, range(len(symbols))))
            neigh2elem.append(np.searchsorted(elem, range(len(elements))))

        self._cache[cutoff_distance] = {
            'distance_vector': distance_vector,
            'distance': distance,
            'neigh2elem': neigh2elem,
            'neigh2j': neigh2j,
            }

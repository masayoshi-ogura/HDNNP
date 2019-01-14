# coding: utf-8

"""Interatomic potential dataset for property of HDNNP. """

import numpy as np

from hdnnpy.dataset.property.property_dataset_base import PropertyDatasetBase


class InteratomicPotentialDataset(PropertyDatasetBase):
    """Interatomic potential dataset for property of HDNNP. """
    PROPERTIES = ['energy', 'force', 'harmonic', 'third_order']
    """list [str]: Names of properties for each derivative order."""
    COEFFICIENTS = [1.0, -1.0]
    """list [float]: Coefficient values of each properties."""
    UNITS = ['eV/atom', 'eV/$\\AA$']
    """list [str]: Units of properties for each derivative order."""
    name = 'interatomic_potential'
    """str: Name of this property class."""

    def __init__(self, order, structures):
        """
        It accepts 0 or 3 for ``order``.

        Notes:
            Currently you cannot use order = 2 or 3, since it is not
            implemented.

        Args:
            order (int): passed to super class.
            structures (list [AtomicStructure]): passed to super class.
        """
        assert 0 <= order <= 3
        super().__init__(order, structures)

    def calculate_properties(self, structure):
        """Calculate required properties for a structure data.

        Args:
            structure (AtomicStructure):
                A structure data to calculate properties.

        Returns:
            list [~numpy.ndarray]: Calculated properties.
            The length is the same as ``order`` given at initialization.
        """
        n_property = 1
        n_deriv = len(structure) * 3
        dataset = []
        if self._order >= 0:
            energy = (self._calculate_energy(structure)
                      .astype(np.float32)
                      .reshape(n_property))
            dataset.append(energy)
        if self._order >= 1:
            force = (self._calculate_force(structure)
                     .astype(np.float32)
                     .reshape(n_property, n_deriv))
            dataset.append(force)
        if self._order >= 2:
            harmonic = (self._calculate_harmonic(structure)
                        .astype(np.float32)
                        .reshape(n_property, n_deriv, n_deriv))
            dataset.append(harmonic)
        if self._order >= 3:
            third_order = (self._calculate_third_order(structure)
                           .astype(np.float32)
                           .reshape(n_property, n_deriv, n_deriv, n_deriv))
            dataset.append(third_order)
        return dataset

    @staticmethod
    def _calculate_energy(structure):
        """Calculate atomic energy."""
        return structure.get_potential_energy() / len(structure)

    @staticmethod
    def _calculate_force(structure):
        """Calculate interatomic forces."""
        return structure.get_forces()

    @staticmethod
    def _calculate_harmonic(structure):
        """Calculate 2nd-order harmonic force constant."""
        raise NotImplementedError

    @staticmethod
    def _calculate_third_order(structure):
        """Calculate 3rd-order anharmonic force constant."""
        raise NotImplementedError

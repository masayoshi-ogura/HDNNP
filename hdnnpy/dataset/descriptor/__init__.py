# coding: utf-8

"""Descriptor dataset subpackage."""

__all__ = [
    'DESCRIPTOR_DATASET',
    ]

from hdnnpy.dataset.descriptor.symmetry_function_dataset import (
    SymmetryFunctionDataset)
from hdnnpy.dataset.descriptor.weighted_symmetry_function_dataset import (
    WeightedSymmetryFunctionDataset)

DESCRIPTOR_DATASET = {
    SymmetryFunctionDataset.name: SymmetryFunctionDataset,
    WeightedSymmetryFunctionDataset.name: WeightedSymmetryFunctionDataset,
    }

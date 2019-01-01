# coding: utf-8

"""Descriptor dataset subpackage."""

__all__ = [
    'DESCRIPTOR_DATASET',
    ]

from hdnnpy.dataset.descriptor.symmetry_function_dataset import (
    SymmetryFunctionDataset)

DESCRIPTOR_DATASET = {
    SymmetryFunctionDataset.name: SymmetryFunctionDataset,
    }

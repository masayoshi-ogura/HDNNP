# coding: utf-8

__all__ = [
    'DESCRIPTOR_DATASET',
    ]

from hdnnpy.dataset.descriptor.symmetry_function_dataset import (
    SymmetryFunctionDataset)

DESCRIPTOR_DATASET = {
    'symmetry_function': SymmetryFunctionDataset,
    }

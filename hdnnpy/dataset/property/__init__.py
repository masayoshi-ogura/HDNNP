# coding: utf-8

"""Property dataset classes and their base class."""

__all__ = [
    'PROPERTY_DATASET',
    ]

from hdnnpy.dataset.property.interatomic_potential_dataset import (
    InteratomicPotentialDataset)

PROPERTY_DATASET = {
    InteratomicPotentialDataset.name: InteratomicPotentialDataset,
    }

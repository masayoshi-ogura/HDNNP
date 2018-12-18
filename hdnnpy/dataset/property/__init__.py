# coding: utf-8

__all__ = [
    'PROPERTY_DATASET',
    ]

from hdnnpy.dataset.property.interatomic_potential_dataset import (
    InteratomicPotentialDataset)

PROPERTY_DATASET = {
    InteratomicPotentialDataset.name: InteratomicPotentialDataset,
    }

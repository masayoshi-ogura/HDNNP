# coding: utf-8

"""Property dataset subpackage."""

__all__ = [
    'PROPERTY_DATASET',
    ]

from hdnnpy.dataset.property.interatomic_potential_dataset import (
    InteratomicPotentialDataset)
from hdnnpy.dataset.property.born_effective_charge_dataset import (
    BornEffectiveChargeDataset)

PROPERTY_DATASET = {
    InteratomicPotentialDataset.name: InteratomicPotentialDataset,
    BornEffectiveChargeDataset.name: BornEffectiveChargeDataset
    }

# coding: utf-8

"""Loss function classses for HDNNP training."""

__all__ = [
    'LOSS_FUNCTION',
    ]

from hdnnpy.training.loss_function.first import First
from hdnnpy.training.loss_function.potential import Potential
from hdnnpy.training.loss_function.zeroth import Zeroth

LOSS_FUNCTION = {
    First.name: First,
    Potential.name: Potential,
    Zeroth.name: Zeroth,
    }

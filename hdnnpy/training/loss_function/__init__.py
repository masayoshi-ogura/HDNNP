# coding: utf-8

"""Loss function classses for HDNNP training."""

__all__ = [
    'LOSS_FUNCTION',
    ]

from hdnnpy.training.loss_function.zeroth import Zeroth
from hdnnpy.training.loss_function.first import First
from hdnnpy.training.loss_function.mix import Mix
# from hdnnpy.training.loss_function.potential import Potential

LOSS_FUNCTION = {
    Zeroth.name: Zeroth,
    First.name: First,
    Mix.name: Mix,
    # Potential.name: Potential,
    }

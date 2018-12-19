# coding: utf-8

__all__ = [
    'LOSS_FUNCTION',
    'Evaluator',
    'Manager',
    'Updater',
    'ScatterPlot',
    'set_log_scale',
    ]

from hdnnpy.training.extensions import (Evaluator,
                                        ScatterPlot,
                                        set_log_scale,
                                        )
from hdnnpy.training.loss_functions import (first_only,
                                            mix,
                                            zeroth_only,
                                            )
from hdnnpy.training.manager import Manager
from hdnnpy.training.updater import Updater


LOSS_FUNCTION = {
    first_only.__name__: first_only,
    mix.__name__: mix,
    zeroth_only.__name__: zeroth_only,
    }

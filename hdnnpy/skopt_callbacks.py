# -*- coding: utf-8 -*-

from skopt.callbacks import EarlyStopper


class SamePointStopper(EarlyStopper):
    def __init__(self):
        super(EarlyStopper, self).__init__()

    def _criterion(self, result):
        if len(result.x_iters) < 2:
            return None

        last_x = result.x_iters[-1]
        min_delta_x = min([result.space.distance(last_x, xi)
                           for xi in result.x_iters[:-1]])
        return abs(min_delta_x) <= 1e-8  # same criterion with UserWarning

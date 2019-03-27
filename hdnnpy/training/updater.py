# coding: utf-8

"""Updater for HDNNP training."""

import chainer


class Updater(chainer.training.updaters.StandardUpdater):
    """Updater for HDNNP training using `HighDimensionalNNP` and
    `MasterNNP`."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_core(self):
        """Calculate gradient of parameters using `HighDimensionalNNP`
        and collect them in `MasterNNP` and update parameters."""
        master_opt = self.get_optimizer('master')
        main_opt = self.get_optimizer('main')
        master_nnp = master_opt.target
        hdnnp = main_opt.target

        batch = self.converter(self.get_iterator('main').next(), self.device)

        master_nnp.cleargrads()
        hdnnp.cleargrads()

        loss = self.loss_func(**batch)
        loss.backward()

        hdnnp.reduce_grad_to(master_nnp)
        master_opt.update()
        hdnnp.sync_param_with(master_nnp)

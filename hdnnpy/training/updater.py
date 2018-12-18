# coding: utf-8

import chainer


class Updater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update_core(self):
        master_opt = self.get_optimizer('master')
        main_opt = self.get_optimizer('main')
        masters = master_opt.target
        hdnnp = main_opt.target

        batch = self.converter(self.get_iterator('main').next(), self.device)
        half = len(batch) // 2
        inputs, labels = batch[:half], batch[half:]

        masters.cleargrads()
        hdnnp.cleargrads()

        loss = self.loss_func(inputs, labels, train=True)
        loss.backward()

        hdnnp.reduce_grad_to(masters)
        master_opt.update()
        hdnnp.sync_param_with(masters)

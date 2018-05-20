# -*- coding: utf-8 -*-

import chainer


class HDUpdater(chainer.training.updaters.StandardUpdater):
    def __init__(self, *args, **kwargs):
        super(HDUpdater, self).__init__(*args, **kwargs)

    def update_core(self):
        master_opt = self.get_optimizer('master')
        main_opt = self.get_optimizer('main')
        masters = master_opt.target
        hdnnp = main_opt.target

        batch = self.converter(self.get_iterator('main').next(), self.device)

        masters.cleargrads()
        hdnnp.cleargrads()

        _, _, loss = hdnnp(*batch, train=True)
        loss.backward()

        hdnnp.reduce_grad_to(masters)
        master_opt.update()
        hdnnp.sync_param_with(masters)

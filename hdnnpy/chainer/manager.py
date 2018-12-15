# coding: utf-8

__all__ = [
    'Manager',
    ]

from contextlib import AbstractContextManager
import pickle
import signal

import chainer
import numpy as np

from hdnnpy.utils import (MPI,
                          pprint,
                          )


class Manager(AbstractContextManager):
    def __init__(self, tag, trainer, result, is_snapshot):
        self._result = result
        self._signum = None
        self._is_snapshot = is_snapshot
        self._tag = tag
        self._trainer = trainer

    def __enter__(self):
        self._old_sigint_handler = signal.signal(
            signal.SIGINT, self._snapshot)
        self._old_sigterm_handler = signal.signal(
            signal.SIGTERM, self._snapshot)
        MPI.comm.Barrier()

    def __exit__(self, type_, value, traceback):
        signal.signal(signal.SIGINT, self._old_sigint_handler)
        signal.signal(signal.SIGTERM, self._old_sigterm_handler)
        if not self._signum:
            self._result['training_time'] += self._trainer.elapsed_time
            observation = {
                k: v.data.item() if isinstance(v, chainer.Variable)
                else v.item() if isinstance(v, np.float64)
                else v
                for k, v in self._trainer.observation.items()}
            self._result['observation'].append(
                {'tag': self._tag, **observation})

    def _snapshot(self, signum, _):
        self._signum = signal.Signals(signum)
        if self._is_snapshot and MPI.rank == 0:
            pprint(f'Stop {self._tag} training by signal:'
                   f' {self._signum.name}!\n'
                   f'Take trainer snapshot at epoch:'
                   f' {self._trainer.updater.epoch}')
            chainer.serializers.save_npz(
                self._trainer.out/'trainer_snapshot.npz', self._trainer)
            (self._trainer.out/'interim_result.pickle').write_bytes(
                pickle.dumps(self._result))

        # must raise any Exception to stop trainer.run()
        raise InterruptedError(f'Chainer training loop is interrupted by'
                               f' {self._signum.name}')

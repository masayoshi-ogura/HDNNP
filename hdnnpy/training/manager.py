# coding: utf-8

from contextlib import AbstractContextManager
import pickle
import signal

import chainer
import numpy as np

from hdnnpy.utils import (MPI,
                          pprint,
                          )


class Manager(AbstractContextManager):
    def __init__(
            self, tag, trainer, result, is_snapshot=True):
        self._tag = tag
        self._trainer = trainer
        self._result = result
        self._is_snapshot = is_snapshot
        self._is_allow = True
        self._trainer_snapshot = trainer.out / 'trainer_snapshot.npz'
        self._interim_result = trainer.out / 'interim_result.pickle'
        self._signum = None

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

    def allow_to_run(self):
        return self._is_allow

    def check_resume(self, resume_tag):
        if self._tag == resume_tag:
            self._resume()
            self._is_allow = True
        elif self._trainer_snapshot.exists():
            self._is_allow = False
        else:
            self._is_allow = True

    def _resume(self):
        pprint(f'Resume training loop from dataset tagged "{self._tag}"')
        chainer.serializers.load_npz(self._trainer_snapshot, self._trainer)
        interim_result = pickle.loads(self._interim_result.read_bytes())
        self._result['training_time'] += interim_result['training_time']
        self._result['observation'].extend(interim_result['observation'])
        # remove snapshot
        MPI.comm.Barrier()
        if MPI.rank == 0:
            self._trainer_snapshot.unlink()
            self._interim_result.unlink()

    def _snapshot(self, signum, _):
        self._signum = signal.Signals(signum)
        if self._is_snapshot and MPI.rank == 0:
            pprint(f'Stop {self._tag} training by signal:'
                   f' {self._signum.name}!\n'
                   f'Take trainer snapshot at epoch:'
                   f' {self._trainer.updater.epoch}')
            chainer.serializers.save_npz(self._trainer_snapshot, self._trainer)
            self._interim_result.write_bytes(pickle.dumps(self._result))

        # must raise any Exception to stop trainer.run()
        raise InterruptedError(
            f'Chainer training loop is interrupted by {self._signum.name}')

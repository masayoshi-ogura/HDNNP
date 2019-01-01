# coding: utf-8

"""Context manager to take trainer snapshot and decide whether to train
or not."""

from contextlib import AbstractContextManager
import pickle
import signal

import chainer
import numpy as np

from hdnnpy.utils import (MPI, pprint)


class Manager(AbstractContextManager):
    """Context manager to take trainer snapshot and decide whether to
    train or not."""
    def __init__(self, tag, trainer, result, is_snapshot=True):
        """
        Args:
            tag (str): Tag of dataset used for training.
            trainer (~chainer.training.Trainer):
                Trainer object to be managed.
            result (dict):
                Dictionary object containing total elapsed time and
                metrics value corresponding to the type of loss
                function. Even when training is stopped / resumed, it is
                retained.
            is_snapshot (bool, optional): Take trainer snapshot if True.
        """
        self._tag = tag
        self._trainer = trainer
        self._result = result
        self._is_snapshot = is_snapshot
        self._is_allow = True
        self._trainer_snapshot = trainer.out / 'trainer_snapshot.npz'
        self._interim_result = trainer.out / 'interim_result.pickle'
        self._signum = None

    def __enter__(self):
        """Replace signal handler of SIGINT and SIGTERM."""
        self._old_sigint_handler = signal.signal(
            signal.SIGINT, self._snapshot)
        self._old_sigterm_handler = signal.signal(
            signal.SIGTERM, self._snapshot)

    def __exit__(self, type_, value, traceback):
        """Restore signal handler of SIGINT and SIGTERM, and record the
        result of training."""
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

    @property
    def allow_to_run(self):
        """Whether the given trainer can train with the dataset."""
        return self._is_allow

    def check_to_resume(self, resume_tag):
        """Decide whether to train or not.

        If current tag of dataset is equal to ``resume_tag``, restore
        the state of trainer from snapshot file.

        Args:
            resume_tag (str):
                Tag of dataset when snapshot was taken last time.
        """
        if self._tag == resume_tag:
            self._resume()
            self._is_allow = True
        elif self._trainer_snapshot.exists():
            self._is_allow = False
        else:
            self._is_allow = True

    def _resume(self):
        """Restore the state of trainer from snapshot file."""
        pprint(f'Resume training loop from dataset tagged "{self._tag}"')
        chainer.serializers.load_npz(self._trainer_snapshot, self._trainer)
        interim_result = pickle.loads(self._interim_result.read_bytes())
        self._result['training_time'] += interim_result['training_time']
        self._result['observation'].extend(interim_result['observation'])
        # remove snapshot
        if MPI.rank != 0:
            return

        self._trainer_snapshot.unlink()
        self._interim_result.unlink()

    def _snapshot(self, signum, _):
        """Take trainer snapshot."""
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

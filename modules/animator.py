from os import path
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

from config import hp
from config import file_


def update(i, ax, pred, true):
    if i != 0:
        ax.cla()

    ax.scatter(pred[i], true)
    ax.set_title('epochs={}'.format(i+1))
    min = np.min(true)
    max = np.max(true)
    ax.set_xlim([min, max])
    ax.set_ylim([min, max])


class Animator(object):
    def __init__(self, natom, nsample):
        nfigure = 1 + 3 * natom
        self.natom = natom
        self.preds = np.empty((hp.nepoch, nsample, nfigure))
        self.true = np.empty((nsample, nfigure))

    def set_pred(self, m, E_pred, F_pred):
        # nepoch x nsample x 1+3*natom -> 1+3*natom x nepoch x nsample
        self.preds[m] = np.c_[E_pred, F_pred.reshape((-1, 3 * self.natom))]

    def set_true(self, E_true, F_true):
        # nsample x 1+3*natom -> 1+3*natom x nsample
        self.true = np.c_[E_true, F_true.reshape((-1, 3 * self.natom))]

    def save_fig(self, ext):
        self.preds = self.preds.transpose(2, 0, 1)
        self.true = self.true.transpose(1, 0)

        for i, (pred, true) in enumerate(zip(self.preds, self.true)):
            if i == 0:
                filename = 'energy.{}'.format(ext)
            elif i % 3 == 0:
                filename = 'force_{}x.{}'.format((i-1)/3+1, ext)
            elif i % 3 == 1:
                filename = 'force_{}y.{}'.format((i-1)/3+1, ext)
            elif i % 3 == 2:
                filename = 'force_{}z.{}'.format((i-1)/3+1, ext)

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if ext == 'gif':
                anime = FuncAnimation(fig, update, fargs=(ax, pred, true), interval=100, frames=hp.nepoch)
                anime.save(path.join(file_.fig_dir, filename), writer='imagemagick')
            elif ext == 'png':
                update(0, ax, pred, true)
                fig.savefig(path.join(file_.fig_dir, filename))
            plt.close(fig)

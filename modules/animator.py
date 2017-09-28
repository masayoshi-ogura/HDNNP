from os import path
from os import mkdir
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
        self.natom = natom
        self.preds = {'energy': np.empty((hp.nepoch, nsample)), 'force': np.empty((hp.nepoch, nsample * 3*natom))}
        self.true = {'energy': np.empty(nsample), 'force': np.empty((nsample * 3*natom))}

    def set_pred(self, m, E_pred, F_pred):
        self.preds['energy'][m] = E_pred.reshape(-1)
        self.preds['force'][m] = F_pred.reshape(-1)

    def set_true(self, E_true, F_true):
        self.true['energy'] = E_true.reshape(-1)
        self.true['force'] = F_true.reshape(-1)

    def save_fig(self, datestr, config, ext):
        plt.ioff()
        for s in ['energy', 'force']:
            save_dir = path.join(file_.fig_dir, datestr)
            if not path.exists(save_dir):
                mkdir(save_dir)
            file = path.join(save_dir, '{}-{}.{}'.format(config, s, ext))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            if ext == 'gif':
                anime = FuncAnimation(fig, update, fargs=(ax, self.preds[s], self.true[s]), interval=100, frames=hp.nepoch)
                anime.save(file, writer='imagemagick')
            elif ext == 'png':
                update(0, ax, self.preds[s], self.true[s])
                fig.savefig(file)
            plt.close(fig)

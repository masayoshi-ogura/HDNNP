from os import path
from os import mkdir
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from collections import defaultdict

from config import hp
from config import bool_
from config import file_


class Animator(object):
    def __init__(self, type):
        self._preds = defaultdict(list)
        self._true = {}
        self._type = type

    @property
    def preds(self):
        return self._preds

    @preds.setter
    def preds(self, pred):
        self._preds['energy'].append(pred[0].reshape(-1))
        self._preds['force'].append(pred[1].reshape(-1))

    @property
    def true(self):
        return self._true

    @true.setter
    def true(self, true):
        self._true['energy'] = true[0].reshape(-1)
        self._true['force'] = true[1].reshape(-1)

    def save_fig(self, datestr, config):
        plt.ioff()
        for s in ['energy', 'force']:
            save_dir = path.join(file_.fig_dir, datestr)
            if not path.exists(save_dir):
                mkdir(save_dir)
            min = np.min(self.true[s])
            max = np.max(self.true[s])

            if bool_.SAVE_GIF:
                file = path.join(save_dir, '{}-{}-{}.gif'.format(config, self._type, s))
                fig = plt.figure()
                artists = [self.artist(i+1, self.preds[s][i], self.true[s], min, max) for i in range(hp.nepoch)]
                anime = ArtistAnimation(fig, artists, interval=50, blit=True)
                anime.save(file, writer='imagemagick')
                plt.close(fig)

            file = path.join(save_dir, '{}-{}-{}.png'.format(config, self._type, s))
            fig = plt.figure()
            self.artist(hp.nepoch, self.preds[s][-1], self.true[s], min, max)
            fig.savefig(file)
            plt.close(fig)

    def artist(self, i, pred, true, min, max):
        artist = [plt.scatter(pred, true, c='blue'),
                  plt.xlim(min, max),
                  plt.ylim(min, max),
                  plt.text(0.5, 0.9, 'epochs={}'.format(i), fontsize=12.0, ha='center', transform=plt.gcf().transFigure)]
        return artist

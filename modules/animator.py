from os import path
from os import mkdir
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
from collections import defaultdict
import networkx as nx
from itertools import product

from config import hp
from config import bool_
from config import file_
from config import visual


class Animator(object):
    def __init__(self):
        self._preds = defaultdict(list)
        self._true = {}

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

    def save(self, datestr, config, type):
        plt.ioff()
        for s in ['energy', 'force']:
            save_dir = path.join(file_.fig_dir, datestr)
            if not path.exists(save_dir):
                mkdir(save_dir)
            min = np.min(self._true[s])
            max = np.max(self._true[s])

            if bool_.SAVE_GIF:
                file = path.join(save_dir, '{}-{}-{}.gif'.format(config, type, s))
                fig = plt.figure()
                artists = [self._artist(i+1, self._preds[s][i], self._true[s], min, max) for i in xrange(hp.nepoch)]
                anime = ArtistAnimation(fig, artists, interval=50, blit=True)
                anime.save(file, writer='imagemagick')
                plt.close(fig)

            file = path.join(save_dir, '{}-{}-{}.png'.format(config, type, s))
            fig = plt.figure()
            self._artist(hp.nepoch, self._preds[s][-1], self._true[s], min, max)
            fig.savefig(file)
            plt.close(fig)

    def _artist(self, i, pred, true, min, max):
        artist = [plt.scatter(pred, true, c='blue'),
                  plt.xlim(min, max),
                  plt.ylim(min, max),
                  plt.text(0.5, 0.9, 'epochs={}'.format(i), fontsize=12.0, ha='center', transform=plt.gcf().transFigure)]
        return artist


def visualize_SingleNNP(nnp, subdir):
    nlayer = len(nnp.shape)
    ymax = float(max(nnp.shape))

    G = nx.Graph()
    pos = {}
    weight = []
    bias = []

    # nodes & pos
    cum = 0
    for i, node in enumerate(nnp.shape):
        if i < nlayer - 1:
            G.add_nodes_from(range(cum, cum+node+1))
            x = i+1
            for j in range(node):
                y = ymax * (j+1) / (node+1)
                pos[cum+j] = (x, y)
            y = - ymax / 5
            pos[cum+node] = (x, y)
            weight.extend(range(cum, cum+node))
            bias.append(cum+node)
            cum += node + 1
        else:
            G.add_nodes_from(range(cum, cum+node))
            x = i+1
            for j in range(node):
                y = ymax * (j+1) / (node+1)
                pos[cum+j] = (x, y)
            y = - ymax / 5
            pos[cum+node] = (x, y)
            weight.extend(range(cum, cum+node))

    # edges
    cum = 0
    for p in nnp.params:
        if p.ndim == 2:
            G.add_edges_from(product(range(cum, cum+p.shape[0]), range(cum+p.shape[0]+1, cum+p.shape[0]+1+p.shape[1])))
            cum += p.shape[0]
        elif p.ndim == 1:
            G.add_edges_from(product([cum], range(cum+1, cum+1+p.shape[0])))
            cum += 1
    params = np.concatenate([p.flatten() for p in nnp.params]).ravel()

    # labels
    labels = {'bias': 'bias', 'input': 'input', 'output': 'output'}
    pos['bias'] = (0.5, - ymax / 5)
    pos['input'] = (0.5, ymax / 2)
    pos['output'] = (nlayer+0.5, ymax / 2)

    plt.clf()
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=50,
                           nodelist=weight,
                           node_color='red',
                           node_shape='o')
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=50,
                           nodelist=bias,
                           node_color='blue',
                           node_shape='s')
    edges = nx.draw_networkx_edges(G,
                                   pos,
                                   width=4,
                                   edge_color=params,
                                   edge_cmap=plt.cm.get_cmap(visual.cmap),
                                   edge_vmin=visual.vmin,
                                   edge_vmax=visual.vmax)
    nx.draw_networkx_labels(G, pos, labels, font_size=16)
    plt.colorbar(edges)
    plt.xlim(-0, nlayer+1)
    plt.savefig(path.join(file_.fig_dir, subdir, 'network.png'))

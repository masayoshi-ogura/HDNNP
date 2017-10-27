from config import hp
from config import bool_
from config import file_
from config import visual

from os import path
from os import listdir
from glob import iglob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import ArtistAnimation
import networkx as nx
from itertools import product

from model import SingleNNP

plt.ioff()


def visualize_network(datestr):
    savedmodel = path.join(file_.save_dir, datestr)
    if not path.exists(savedmodel):
        raise

    print 'visualize the network of {}'.format(savedmodel)
    print 'range of coloring is [{} : {}]'.format(visual.vmin, visual.vmax)
    print 'color map: {}'.format(visual.cmap)

    nnp = SingleNNP(1, has_optimizer=False)
    if nnp.load(savedmodel):
        print 'the saved model is SingleNNP.'
        visualize_SingleNNP(nnp, datestr)
    else:
        print 'the saved model is HDNNP.'
        for x in listdir(savedmodel):
            if nnp.load(path.join(savedmodel, x)):
                visualize_SingleNNP(nnp, datestr, x)


def visualize_SingleNNP(nnp, datestr, element=None):
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
    basename = '{}_network.png'.format(element) if element else 'network.png'
    plt.savefig(path.join(file_.fig_dir, datestr, basename))


def visualize_correlation_scatter(datestr):
    def artist(i, pred, true, min, max):
        artist = [plt.scatter(pred, true, c='blue'),
                  plt.xlim(min, max),
                  plt.ylim(min, max),
                  plt.text(0.5, 0.9, 'epochs={}'.format(i), fontsize=12.0, ha='center', transform=plt.gcf().transFigure)]
        return artist

    out_dir = path.join(file_.out_dir, datestr)
    fig_dir = path.join(file_.fig_dir, datestr)
    if not path.exists(out_dir):
        raise

    for output_file in iglob(path.join(out_dir, '*.npz')):
        config = path.basename(output_file).split('.')[0]
        output_data = np.load(output_file)
        print 'visualize the correlation scatter of {}'.format(output_file)

        for key, ndarray in output_data.items():
            min = np.min(ndarray[0])
            max = np.max(ndarray[0])
            # gif
            if bool_.SAVE_GIF:
                fig = plt.figure()
                artists = [artist(i+1, ndarray[i+1], ndarray[0], min, max) for i in xrange(hp.nepoch)]
                anime = ArtistAnimation(fig, artists, interval=50, blit=True)
                anime.save(path.join(fig_dir, '{}_{}.gif'.format(config, key)), writer='imagemagick')
                plt.close(fig)
            # png
            fig = plt.figure()
            artist(hp.nepoch, ndarray[-1], ndarray[0], min, max)
            fig.savefig(path.join(fig_dir, '{}_{}.png'.format(config, key)))
            plt.close(fig)

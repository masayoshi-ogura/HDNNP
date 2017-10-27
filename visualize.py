# -*- coding: utf -*-

from config import file_
from config import visual

from os import path
from sys import argv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import product

from modules.model import SingleNNP
# from modules.model import HDNNP

savedmodel = path.join(file_.save_dir, argv[1])
print 'visualize the network of {}'.format(savedmodel)
print 'range of coloring is [{} : {}]'.format(visual.vmin, visual.vmax)
cmap = plt.cm.RdBu
print 'color map: {}'.format(cmap.name)

nnp = SingleNNP(1)
nnp.load(savedmodel)
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

plt.figure(figsize=(15, 10))
nodes = nx.draw_networkx_nodes(G, pos, node_size=50, nodelist=weight, node_color='red', node_shape='o')
nodes = nx.draw_networkx_nodes(G, pos, node_size=50, nodelist=bias, node_color='blue', node_shape='s')
edges = nx.draw_networkx_edges(G, pos, width=4, edge_color=params, edge_cmap=cmap, edge_vmin=visual.vmin, edge_vmax=visual.vmax)
nx.draw_networkx_labels(G, pos, labels, font_size=16)
plt.colorbar(edges)
plt.xlim(-0, nlayer+1)
plt.savefig(path.join(file_.fig_dir, argv[1], 'network.png'))

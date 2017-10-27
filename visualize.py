# -*- coding: utf -*-

from config import file_
from config import visual

from os import path
from os import listdir
from sys import argv

from modules.model import SingleNNP
from modules.animator import visualize_SingleNNP

savedmodel = path.join(file_.save_dir, argv[1])
if not path.exists(savedmodel):
    raise

print 'visualize the network of {}'.format(savedmodel)
print 'range of coloring is [{} : {}]'.format(visual.vmin, visual.vmax)
print 'color map: {}'.format(visual.cmap)

nnp = SingleNNP(1)
if nnp.load(savedmodel):
    visualize_SingleNNP(nnp, argv[1])
else:
    for x in listdir(savedmodel):
        if nnp.load(path.join(savedmodel, x)):
            visualize_SingleNNP(nnp, path.join(argv[1], x))

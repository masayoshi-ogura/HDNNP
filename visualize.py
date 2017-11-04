# -*- coding: utf -*-

from config import file_

from sys import argv
from os import path
from os import mkdir
from glob import glob

from modules.animator import visualize_network
from modules.animator import visualize_correlation_scatter

if len(argv) > 1:
    datestr = argv[1]
else:
    datestr = glob('output/[0-9]*-[0-9]*')[-1].split('/')[1]

fig_dir = path.join(file_.fig_dir, datestr)
if not path.exists(fig_dir):
    mkdir(fig_dir)
visualize_network(datestr)
visualize_correlation_scatter(datestr)

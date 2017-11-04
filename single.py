# -*- coding: utf-8 -*-

# define variables
from config import file_

# import python modules
from sys import argv
from os import path
from os import makedirs
from shutil import copy2
from datetime import datetime
from time import time

# import own modules
from modules.data import FunctionData
from modules.model import SingleNNP

start = time()
datestr = datetime.now().strftime('%m%d-%H%M%S')
save_dir = path.join(file_.save_dir, datestr)
out_dir = path.join(file_.out_dir, datestr)
makedirs(save_dir)
makedirs(out_dir)
copy2('config.py', path.join(out_dir, 'config.py'))
progress = open(path.join(out_dir, 'progress.dat'), 'w')
output_file = path.join(out_dir, '{}.npz'.format(argv[1]))

dataset = FunctionData(argv[1])
nnp = SingleNNP(dataset.ninput, save_dir)
nnp.fit(dataset, progress=progress)

nnp.save(save_dir, output_file)
progress.write('\n\nTotal time: {}'.format(time()-start))
progress.close()

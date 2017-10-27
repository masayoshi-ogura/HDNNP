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

"""
if argv[1] in ['LJ', 'sin']:
    fig = plt.figure()
    plt.scatter(training_data.input, output_data['training_energy'][0], c='blue')
    plt.scatter(validation_data.input, output_data['validation_energy'][0], c='blue')
    plt.scatter(training_data.input, output_data['training_energy'][-1], c='red')
    plt.scatter(validation_data.input, output_data['validation_energy'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'original_func.png'))
    plt.close(fig)
    fig = plt.figure()
    plt.scatter(training_data.input, output_data['training_force'][0], c='blue')
    plt.scatter(validation_data.input, output_data['validation_force'][0], c='blue')
    plt.scatter(training_data.input, output_data['training_force'][-1], c='red')
    plt.scatter(validation_data.input, output_data['validation_force'][-1], c='yellow')
    fig.savefig(path.join(file_.fig_dir, datestr, 'derivative.png'))
    plt.close()
"""

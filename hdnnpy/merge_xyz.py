# -*- coding: utf-8 -*-
"""Module for merging xyz"""
from pathlib import Path
import ase.io

def merge(args):
    """Function to merges xyz"""

    step_str = args.step
    in_dir = args.input
    output = args.output

    step = int(step_str)

    for file in Path(in_dir).glob('*.xyz'):
        images = ase.io.read(file, index='::{}'.format(step), format='xyz')
        ase.io.write(output, images, format='xyz', append=True)

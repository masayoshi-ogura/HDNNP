# -*- coding: utf-8 -*-
"""Module for merging xyz"""
from pathlib import Path
import ase.io

def merge(step_str, in_dir, output):
    """Function to merges xyz"""
    step = int(step_str)

    for file in Path(in_dir).glob('*.xyz'):
        images = ase.io.read(file, index='::{}'.format(step), format='xyz')
        ase.io.write(output, images, format='xyz', append=True)

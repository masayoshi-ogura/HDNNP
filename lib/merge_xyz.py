#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
import ase.io

args = sys.argv
step = int(args[1])
in_dir = args[2]
output = args[3]

for f in Path(in_dir).glob('*.xyz'):
    images = ase.io.read(f, index='::{}'.format(step), format='xyz')
    ase.io.write(output, images, format='xyz', append=True)

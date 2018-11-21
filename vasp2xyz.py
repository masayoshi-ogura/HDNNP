#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import ase.io

args = sys.argv

if len(args) is not 3:
    sys.stdout.write("args should have 3 but has %d" % len(args))
    sys.exit(1)

config = args[1]
input = args[2]
output = args[3]

images = []
for atoms in ase.io.iread(input, index=':', format='vasp-out'):
    # stress = atoms.get_stress(voigt=False)
    # atoms.set_param_value('stress', stress)
    atoms.info['config_type'] = config + atoms.get_chemical_formula()
    images.append(atoms)
ase.io.write(output, images, format='xyz')

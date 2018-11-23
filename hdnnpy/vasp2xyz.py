# -*- coding: utf-8 -*-
"""Module to convert vasp output to xyz format"""
import ase.io

def convert(config, outcar_file, output):
    """Function to convert vasp output to xyz format"""
    images = []
    
    for atoms in ase.io.iread(outcar_file, index=':', format='vasp-out'):
        # stress = atoms.get_stress(voigt=False)
        # atoms.set_param_value('stress', stress)
        atoms.info['config_type'] = config + atoms.get_chemical_formula()
        images.append(atoms)
    
    ase.io.write(output, images, format='xyz')

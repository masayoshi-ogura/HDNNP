# -*- coding: utf-8 -*-

from pathlib import Path
from tempfile import NamedTemporaryFile
import ase.io

from ..utils import pprint


def parse_poscars(file_paths, save=False):
    tag_xyz_map = {}
    tag_poscars_map = {}
    for poscar_path in file_paths:
        atoms = ase.io.read(poscar_path, format='vasp')
        tag = atoms.info['tag'] = atoms.get_chemical_formula()
        try:
            xyz_path = tag_xyz_map[tag]
            tag_poscars_map[tag].append(poscar_path)
        except KeyError:
            if save:
                xyz_path = poscar_path.with_name('{}.xyz'.format(tag))
                pprint('Sub dataset tagged as "{}" is saved to {}.'
                       .format(tag, xyz_path))
            else:
                xyz_path = Path(NamedTemporaryFile('w', delete=False).name)
                pprint('Sub dataset tagged as "{}" is temporarily saved to {}.\n'
                       'If ABEND and this file remains, delete it manually.'
                       .format(tag, xyz_path))
            tag_xyz_map[tag] = xyz_path
            tag_poscars_map[tag] = [poscar_path]
        ase.io.write(str(xyz_path), atoms, format='xyz', append=True)
    pprint()
    return tag_xyz_map, tag_poscars_map

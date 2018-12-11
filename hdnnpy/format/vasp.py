# -*- coding: utf-8 -*-

__all__ = [
    'parse_poscars',
    ]

from pathlib import Path
from tempfile import NamedTemporaryFile

import ase.io

from hdnnpy.utils import pprint


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
                xyz_path = poscar_path.with_name(f'{tag}.xyz')
                pprint(f'Sub dataset tagged as "{tag}" is saved to '
                       f'{xyz_path}.')

            else:
                xyz_path = Path(NamedTemporaryFile('w', delete=False).name)
                pprint(f'Sub dataset tagged as "{tag}" is temporarily saved '
                       f'to {xyz_path}.\n'
                       f'If ABEND and this file remains, delete it manually.')
            tag_xyz_map[tag] = xyz_path
            tag_poscars_map[tag] = [poscar_path]
        ase.io.write(str(xyz_path), atoms, format='xyz', append=True)
    pprint()
    return tag_xyz_map, tag_poscars_map

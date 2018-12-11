# -*- coding: utf-8 -*-

from pathlib import Path
from tempfile import NamedTemporaryFile
import ase.io

from ..utils import pprint, mkdir


def parse_xyz(file_path, save=True):
    tag_xyz_map = {}
    tag_file = file_path.with_name(f'{file_path.name}.tag')
    if tag_file.exists():
        for tag in tag_file.read_text().split():
            tag_xyz_map[tag] = (Path(file_path.with_name(tag))
                                / 'Atomic_Structure.xyz')
    else:
        for atoms in ase.io.iread(str(file_path), index=':', format='xyz'):
            tag = atoms.info['tag']
            try:
                xyz_path = tag_xyz_map[tag]
            except KeyError:
                if save:
                    mkdir(file_path.with_name(tag))
                    xyz_path = file_path.with_name(tag)/'Atomic_Structure.xyz'
                    pprint(f'Sub dataset tagged as "{tag}" is saved to '
                           f'{xyz_path}.')

                else:
                    xyz_path = Path(NamedTemporaryFile('w', delete=False).name)
                    pprint(f'Sub dataset tagged as "{tag}" is temporarily '
                           f'saved to {xyz_path}.\n'
                           f'If ABEND and this file remains, delete it '
                           f'manually.')
                tag_xyz_map[tag] = xyz_path
            ase.io.write(str(xyz_path), atoms, format='xyz', append=True)
        if save:
            tag_file.write_text('\n'.join(sorted(tag_xyz_map)) + '\n')
    pprint()
    return tag_xyz_map

# -*- coding: utf-8 -*-

from pathlib import Path
from tempfile import NamedTemporaryFile
import ase.io

from ..utils import pprint, mkdir


def parse_xyz(file_path, save=True):
    tag_xyz_map = {}
    tag_file = file_path.with_name('{}.tag'.format(file_path.name))
    if tag_file.exists():
        for tag in tag_file.read_text().split():
            tag_xyz_map[tag] = Path(file_path.with_name(tag))/'Atomic_Structure.xyz'
    else:
        for atoms in ase.io.iread(str(file_path), index=':', format='xyz'):
            tag = atoms.info['tag']
            try:
                xyz_path = tag_xyz_map[tag]
            except KeyError:
                if save:
                    mkdir(file_path.with_name(tag))
                    xyz_path = file_path.with_name(tag)/'Atomic_Structure.xyz'
                    pprint('Sub dataset tagged as "{}" is saved to {}.'
                           .format(tag, xyz_path))
                else:
                    xyz_path = Path(NamedTemporaryFile('w', delete=False).name)
                    pprint('Sub dataset tagged as "{}" is temporarily saved to {}.\n'
                           'If ABEND and this file remains, delete it manually.'
                           .format(tag, xyz_path))
                tag_xyz_map[tag] = xyz_path
            ase.io.write(str(xyz_path), atoms, format='xyz', append=True)
        if save:
            tag_file.write_text('\n'.join(sorted(tag_xyz_map)) + '\n')
    pprint()
    return tag_xyz_map

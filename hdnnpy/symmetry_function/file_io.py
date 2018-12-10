# -*- coding: utf-8 -*-

import ase.io

from ..util import mkdir


def parse_xyz(file_path):
    tag_file = file_path.with_name('{}.tag'.format(file_path.name))
    if tag_file.exists():
        tags = tag_file.read_text().split()
        return tags
    else:
        tags = set()
        for atoms in ase.io.iread(str(file_path), index=':', format='xyz'):
            tag = atoms.info['tag']
            tags.add(tag)
            tag_dir = file_path.with_name(tag)
            mkdir(tag_dir)
            ase.io.write(str(tag_dir/'Atomic_Structure.xyz'), atoms, format='xyz', append=True)
        tags = sorted(tags)
        tag_file.write_text('\n'.join(tags) + '\n')
        return tags

# coding: utf-8

"""Functions to handle xyz file format."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import ase.io

from hdnnpy.utils import (MPI, mkdir, pprint)


def parse_xyz(file_path, save=True, verbose=True):
    """Parse a xyz format file and bunch structures by the same tag.

    Args:
        file_path (~pathlib.Path): File path to parse.
        save (bool, optional):
            If True, save the structures bunched by the same tag into
            files. Otherwise, save into temporarily files.
        verbose (bool, optional): Print log to stdout.

    Returns:
        tuple: 2-element tuple containing:

        - tag_xyz_map (dict): Tag to file path mapping.
        - elements (list [str]):
            All elements contained in the parsed file.
    """
    tag_xyz_map = {}
    elements = set()

    # non root process
    if MPI.rank != 0:
        tag_xyz_map = MPI.comm.bcast(tag_xyz_map, root=0)
        elements = MPI.comm.bcast(elements, root=0)
        return tag_xyz_map, sorted(elements)

    # root process
    info_file = file_path.with_name(f'{file_path.name}.dat')
    if info_file.exists():
        elements, *tags = info_file.read_text().strip().split('\n')
        elements = set(elements.split())
        for tag in tags:
            tag_xyz_map[tag] = (Path(file_path.with_name(tag))
                                / 'structure.xyz')
    else:
        for atoms in ase.io.iread(str(file_path), index=':', format='xyz'):
            tag = atoms.info['tag']
            try:
                xyz_path = tag_xyz_map[tag]
            except KeyError:
                if save:
                    mkdir(file_path.with_name(tag))
                    xyz_path = file_path.with_name(tag)/'structure.xyz'
                    if verbose:
                        pprint(f'Sub dataset tagged as "{tag}" is saved to'
                               f' {xyz_path}.')

                else:
                    xyz_path = Path(NamedTemporaryFile('w', delete=False).name)
                    if verbose:
                        pprint(f'Sub dataset tagged as "{tag}" is temporarily'
                               f' saved to {xyz_path}.\n'
                               'If ABEND and this file remains, delete it'
                               ' manually.')
                tag_xyz_map[tag] = xyz_path
            ase.io.write(str(xyz_path), atoms, format='xyz', append=True)
            elements.update(atoms.get_chemical_symbols())
        if save:
            info_file.write_text(' '.join(sorted(elements)) + '\n'
                                 + '\n'.join(sorted(tag_xyz_map)) + '\n')

    tag_xyz_map = MPI.comm.bcast(tag_xyz_map, root=0)
    elements = MPI.comm.bcast(elements, root=0)
    return tag_xyz_map, sorted(elements)

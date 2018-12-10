# -*- coding: utf-8 -*-

from collections import defaultdict
import pickle
import ase.io

from .. import settings as stg
from ..util import pprint, mkdir


def load_xyz(xyz_file):
    atoms_list = ase.io.read(str(xyz_file), index=':', format='xyz')
    tag = atoms_list[0].info['tag']
    elemental_composition = atoms_list[0].get_chemical_symbols()
    return atoms_list, tag, elemental_composition


def load_poscar(poscars):
    atoms_list = []
    for poscar in poscars:
        atoms = ase.io.read(str(poscar), format='vasp')
        atoms.info['tag'] = tag = atoms.get_chemical_formula()
        elemental_composition = atoms.get_chemical_symbols()
        atoms_list.append(atoms)
    return atoms_list, tag, elemental_composition


def parse_xyzfile(xyz_file):
    tag_file = xyz_file.with_name('{}.tag'.format(xyz_file.name))
    if tag_file.exists():
        tags = tag_file.read_text().split()
        return tags
    else:
        tags = set()
        for atoms in ase.io.iread(str(xyz_file), index=':', format='xyz'):
            tag = atoms.info['tag']
            tags.add(tag)
            tag_dir = xyz_file.with_name(tag)
            mkdir(tag_dir)
            ase.io.write(str(tag_dir/'structure.xyz'), atoms, format='xyz', append=True)
        tags = sorted(tags)
        tag_file.write_text('\n'.join(tags) + '\n')
        return tags

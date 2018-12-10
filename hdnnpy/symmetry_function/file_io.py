# -*- coding: utf-8 -*-

from collections import defaultdict
import pickle
import ase.io

from .. import settings as stg
from ..util import pprint, mkdir


def load_xyz(xyz_file):
    atoms_list = ase.io.read(str(xyz_file), index=':', format='xyz')
    config = atoms_list[0].info['config_type']
    elemental_composition = atoms_list[0].get_chemical_symbols()
    return atoms_list, config, elemental_composition


def load_poscar(poscars):
    atoms_list = []
    for poscar in poscars:
        atoms = ase.io.read(str(poscar), format='vasp')
        atoms.info['config_type'] = config = atoms.get_chemical_formula()
        elemental_composition = atoms.get_chemical_symbols()
        atoms_list.append(atoms)
    return atoms_list, config, elemental_composition


def parse_xyzfile(xyz_file):
    if stg.mpi.rank == 0:
        pprint('config_type.pickle is not found.\nparsing {} ... '.format(xyz_file), end='')
        config_type = set()
        dataset = defaultdict(list)
        for atoms in ase.io.iread(str(xyz_file), index=':', format='xyz', parallel=False):
            config = atoms.info['config_type']
            config_type.add(config)
            dataset[config].append(atoms)
        xyz_file.with_name('config_type.pickle').write_bytes(pickle.dumps(config_type))

        for config in config_type:
            cfg_dir = xyz_file.with_name(config)
            mkdir(cfg_dir)
            ase.io.write(str(cfg_dir/'structure.xyz'), dataset[config], format='xyz', parallel=False)
        pprint('done')

    stg.mpi.comm.Barrier()

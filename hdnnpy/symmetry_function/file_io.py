from collections import defaultdict
import pickle
import ase.io

from .. import settings as stg
from ..util import pprint, mkdir


def load_xyz(xyz_file):
    atoms = ase.io.read(str(xyz_file), index=':', format='xyz')
    data_dir = xyz_file.parent
    config = data_dir.name
    composition = pickle.loads((data_dir/'composition.pickle').read_bytes())
    return atoms, config, composition


def load_poscar(poscars):
    atoms = [ase.io.read(str(poscar), format='vasp') for poscar in poscars]
    config = atoms[0].get_chemical_formula()
    symbols = atoms[0].get_chemical_symbols()
    composition = {'indices': {k: set([i for i, s in enumerate(symbols) if s == k])
                                     for k in set(symbols)},
                         'atom': symbols,
                         'element': sorted(set(symbols))}
    return atoms, config, composition


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
            composition = {'indices': defaultdict(list), 'atom': [], 'element': []}
            for i, atom in enumerate(dataset[config][0]):
                composition['indices'][atom.symbol].append(i)
                composition['atom'].append(atom.symbol)
            composition['element'] = sorted(set(composition['atom']))

            cfg_dir = xyz_file.with_name(config)
            mkdir(cfg_dir)
            ase.io.write(str(cfg_dir/'structure.xyz'), dataset[config], format='xyz', parallel=False)
            (cfg_dir/'composition.pickle').write_bytes(pickle.dumps(dict(composition)))
        pprint('done')

    stg.mpi.comm.Barrier()

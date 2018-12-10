# -*- coding: utf-8 -*-

from tempfile import NamedTemporaryFile
from collections import defaultdict
from pathlib import Path
import ase.io

from ..settings import stg
from ..util import pprint, mkdir
from .dataset import SymmetryFunctionDataset
from .utils import scatter_dataset


def load_xyz(file_path, preproc):
    file_path = file_path.absolute()
    all_tags = parse_xyz(file_path)
    included_tags = all_tags if 'all' in stg.dataset.tag else stg.dataset.tag

    datasets = []
    elements = set()
    stg.dataset.nsample = 0
    for tag in included_tags:
        if tag not in all_tags:
            pprint('Data tagged "{}" does not exist in {}. Skipped.\n'.format(tag, file_path))
            continue
        pprint('Construct dataset tagged "{}"'.format(tag))

        xyz = file_path.with_name(tag)/'Atomic_Structure.xyz'
        dataset = SymmetryFunctionDataset()
        dataset.load(xyz, verbose=True)
        dataset.save()
        elements.update(dataset.elements)
        stg.dataset.nsample += dataset.nsample

        preproc.decompose(dataset)
        dataset = scatter_dataset(dataset)
        datasets.append(dataset)
        pprint('')

    return datasets, sorted(elements)


def load_poscars(file_paths, preproc):
    tag_poscars_map = defaultdict(list)
    for poscar in file_paths:
        tag = ase.io.read(poscar, format='vasp').get_chemical_formula()
        tag_poscars_map[tag].append(poscar)

    datasets = []
    elements = set()
    for poscars in tag_poscars_map.values():
        with NamedTemporaryFile('w') as xyz:
            for poscar in poscars:
                atoms = ase.io.read(poscar, format='vasp')
                atoms.info['tag'] = atoms.get_chemical_symbols()
                ase.io.write(xyz.name, atoms, format='xyz', append=True)
            dataset = SymmetryFunctionDataset()
            dataset.load(Path(xyz.name), verbose=False)
        preproc.decompose(dataset)
        datasets.append(dataset)
        elements.update(dataset.elements)

    return tag_poscars_map.values(), datasets, sorted(elements)


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

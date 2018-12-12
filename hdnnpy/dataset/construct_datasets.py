# -*- coding: utf-8 -*-

__all__ = [
    'construct_test_datasets',
    'construct_training_datasets',
    ]

import ase.io

from hdnnpy.dataset.hdnnp_dataset import HDNNPDataset
from hdnnpy.settings import stg
from hdnnpy.utils import pprint


def construct_training_datasets(tag_xyz_map, preproc):
    if 'all' in stg.dataset.tag:
        included_tags = sorted(tag_xyz_map)
    else:
        included_tags = stg.dataset.tag

    params = {
        'Rc': stg.dataset.Rc,
        'eta': stg.dataset.eta,
        'Rs': stg.dataset.Rs,
        'lambda_': stg.dataset.lambda_,
        'zeta': stg.dataset.zeta,
        }

    datasets = []
    elements = set()
    stg.dataset.nsample = 0
    for tag in included_tags:
        try:
            xyz_path = tag_xyz_map[tag]
        except KeyError:
            pprint(f'Sub dataset tagged as "{tag}" does not exist. Skipped.')
            continue

        pprint(f'Construct sub dataset tagged as "{tag}"')
        dataset = HDNNPDataset(descriptor='symmetry_function',
                               property_='interatomic_potential',
                               order=1)
        atoms = ase.io.read(xyz_path, index=':', format='xyz')

        descriptor_npz_path = xyz_path.with_name('Symmetry_Function.npz')
        if descriptor_npz_path.exists():
            dataset.descriptor_dataset.load(descriptor_npz_path, verbose=True)
        else:
            dataset.descriptor_dataset.make(atoms, params, verbose=True)
            dataset.descriptor_dataset.save(descriptor_npz_path, verbose=True)

        property_npz_path = xyz_path.with_name('Interatomic_Potential.npz')
        if property_npz_path.exists():
            dataset.property_dataset.load(property_npz_path, verbose=True)
        else:
            dataset.property_dataset.make(atoms, verbose=True)
            dataset.property_dataset.save(property_npz_path, verbose=True)

        dataset.construct(preproc, shuffle=True)

        dataset.scatter()
        datasets.append(dataset)
        elements.update(dataset.elements)
        stg.dataset.nsample += dataset.total_size
    pprint()
    return datasets, elements


def construct_test_datasets(tag_xyz_map, preproc):
    params = {
        'Rc': stg.dataset.Rc,
        'eta': stg.dataset.eta,
        'Rs': stg.dataset.Rs,
        'lambda_': stg.dataset.lambda_,
        'zeta': stg.dataset.zeta,
        }

    datasets = []
    elements = set()
    for xyz_path in tag_xyz_map.values():
        dataset = HDNNPDataset(descriptor='symmetry_function',
                               order=1)
        atoms = ase.io.read(xyz_path, index=':', format='xyz')

        descriptor_npz_path = xyz_path.with_name('Symmetry_Function.npz')
        if descriptor_npz_path.exists():
            dataset.descriptor_dataset.load(descriptor_npz_path, verbose=False)
        else:
            dataset.descriptor_dataset.make(atoms, params, verbose=False)
            dataset.descriptor_dataset.save(descriptor_npz_path, verbose=False)

        dataset.construct(preproc, shuffle=False)

        datasets.append(dataset)
        elements.update(dataset.elements)

    return datasets, elements

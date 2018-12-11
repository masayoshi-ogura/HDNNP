# -*- coding: utf-8 -*-

from pathlib import Path

from .scatter_dataset import scatter_dataset
from .symmetry_function_dataset import SymmetryFunctionDataset
from ..settings import stg
from ..utils import pprint


def construct_training_datasets(tag_xyz_map, preproc):
    included_tags = sorted(tag_xyz_map) if 'all' in stg.dataset.tag else stg.dataset.tag

    datasets = []
    elements = set()
    stg.dataset.nsample = 0
    for tag in included_tags:
        try:
            xyz_path = tag_xyz_map[tag]
            pprint('Construct sub dataset tagged as "{}"'.format(tag))
            dataset = SymmetryFunctionDataset()
            dataset.load(xyz_path, verbose=True)
            dataset.save()
            stg.dataset.nsample += dataset.nsample

            preproc.decompose(dataset)
            dataset = scatter_dataset(dataset)
            datasets.append(dataset)
            elements.update(dataset.elements)
        except KeyError:
            pprint('Sub dataset tagged as "{}" does not exist. Skipped.'.format(tag))
    pprint()
    return datasets, elements


def construct_test_datasets(tag_xyz_map, preproc):
    datasets = []
    elements = set()
    for tag, xyz_path in tag_xyz_map.items():
        dataset = SymmetryFunctionDataset()
        dataset.load(xyz_path, verbose=False)
        preproc.decompose(dataset)
        # dataset = scatter_dataset(dataset)
        datasets.append(dataset)
        elements.update(dataset.elements)

    return datasets, elements

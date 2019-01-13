# coding=utf-8

import fnmatch
import shutil

import chainer
import numpy as np
from traitlets import (Bool, Dict, List, Unicode)
from traitlets.config import Application
import yaml

from hdnnpy.cli.configurables import (
    DatasetConfig, ModelConfig, Path, PredictionConfig,
    )
from hdnnpy.dataset import (AtomicStructure, DatasetGenerator, HDNNPDataset)
from hdnnpy.dataset.descriptor import DESCRIPTOR_DATASET
from hdnnpy.dataset.property import PROPERTY_DATASET
from hdnnpy.format import parse_xyz
from hdnnpy.model import (HighDimensionalNNP, MasterNNP)
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.utils import (MPI, pprint, pyyaml_path_constructor)


class PredictionApplication(Application):
    name = Unicode(u'hdnnpy predict')
    description = ('Predict properties for atomic structures using trained'
                   ' HDNNP.')

    verbose = Bool(
        False,
        help='Set verbose mode'
        ).tag(config=True)

    classes = List([PredictionConfig])

    config_file = Path(
        'prediction_config.py',
        help='Load this config file')

    aliases = Dict({
        'log_level': 'Application.log_level',
        })

    flags = Dict({
        'verbose': ({
            'PredictionApplication': {
                'verbose': True,
                },
            }, 'Set verbose mode'),
        'v': ({
            'PredictionApplication': {
                'verbose': True,
                },
            }, 'Set verbose mode'),
        'debug': ({
            'Application': {
                'log_level': 10,
                },
            }, 'Set log level to DEBUG'),
        })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_config = None
        self.model_config = None
        self.prediction_config = None

    def initialize(self, argv=None):
        self.parse_command_line(argv)
        self.load_config_file(self.config_file)
        self.prediction_config = PredictionConfig(config=self.config)

        yaml.add_constructor('Path', pyyaml_path_constructor)
        training_result = yaml.load(
            (self.prediction_config.load_dir / 'training_result.yaml').open())
        self.dataset_config = DatasetConfig(**training_result['dataset'])
        self.model_config = ModelConfig(**training_result['model'])

    def start(self):
        pc = self.prediction_config
        shutil.copy(self.config_file, pc.load_dir / self.config_file.name)
        tag_xyz_map, pc.elements = parse_xyz(
            pc.data_file, save=False, verbose=self.verbose)
        datasets = self.construct_datasets(tag_xyz_map)
        datasets = DatasetGenerator(*datasets).all()
        results = self.predict(datasets)
        self.dump_result(results)

    def construct_datasets(self, tag_xyz_map):
        dc = self.dataset_config
        pc = self.prediction_config

        preprocesses = []
        for (name, args, kwargs) in dc.preprocesses:
            preprocess = PREPROCESS[name](*args, **kwargs)
            preprocess.load(
                pc.load_dir / 'preprocess' / f'{name}.npz',
                verbose=self.verbose)
            preprocesses.append(preprocess)

        datasets = []
        for pattern in pc.tags:
            for tag in fnmatch.filter(tag_xyz_map, pattern):
                if self.verbose:
                    pprint(f'Construct sub dataset tagged as "{tag}"')
                tagged_xyz = tag_xyz_map.pop(tag)
                structures = AtomicStructure.read_xyz(tagged_xyz)

                # prepare descriptor dataset
                descriptor = DESCRIPTOR_DATASET[dc.descriptor](
                    pc.order, structures, **dc.parameters)
                descriptor.make(verbose=self.verbose)

                # prepare empty property dataset
                property_ = PROPERTY_DATASET[dc.property_](
                    pc.order, structures)

                # construct test dataset from descriptor & property datasets
                dataset = HDNNPDataset(descriptor, property_)
                dataset.construct(
                    all_elements=pc.elements, preprocesses=preprocesses,
                    shuffle=False, verbose=self.verbose)
                datasets.append(dataset)
                dc.n_sample += dataset.total_size

        return datasets

    def predict(self, datasets):
        mc = self.model_config
        pc = self.prediction_config
        results = []

        # master model
        master_nnp = MasterNNP(pc.elements, mc.layers)
        chainer.serializers.load_npz(
            pc.load_dir / 'master_nnp.npz', master_nnp)

        for dataset in datasets:
            # hdnnp model
            hdnnp = HighDimensionalNNP(
                dataset.elemental_composition, mc.layers, pc.order)
            hdnnp.sync_param_with(master_nnp)

            with chainer.using_config('train', False), \
                 chainer.using_config('enable_backprop', False):
                predictions = hdnnp.predict(
                    chainer.dataset.concat_examples(dataset))
            result = {
                **{'tag': dataset.tag},
                **{property_: coefficient * prediction.data
                   for property_, coefficient, prediction
                   in zip(dataset.property.properties,
                          dataset.property.coefficients,
                          predictions)},
                }
            results.append(result)
        return results

    def dump_result(self, results):
        if MPI.rank != 0:
            return

        pc = self.prediction_config
        result_file = pc.load_dir / f'prediction_result{pc.dump_format}'
        if pc.dump_format == '.npz':
            kv_result = {}
            for result in results:
                tag = result.pop('tag')
                kv_result.update({tag + '/' + key: value
                                  for key, value in result.items()})
            np.savez(result_file, **kv_result)


def generate_config_file():
    prediction_app = PredictionApplication()
    prediction_app.config_file.write_text(
        prediction_app.generate_config_file())

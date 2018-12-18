# coding=utf-8

import shutil

import ase.io
import chainer
import numpy as np
from traitlets import (Bool, Dict, List, Unicode)
from traitlets.config import Application

from hdnnpy.cli.configurables import (
    DatasetConfig, ModelConfig, Path, PredictionConfig,
    )
from hdnnpy.cli.train import TrainingApplication
from hdnnpy.dataset import (AtomicStructure, DatasetGenerator, HDNNPDataset)
from hdnnpy.format import parse_xyz
from hdnnpy.model import (HighDimensionalNNP, MasterNNP)
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.utils import pprint


class PredictionApplication(Application):
    name = Unicode(u'hdnnpy predict')
    description = ('Predict properties for atomic structures using trained'
                   ' HDNNP.')

    verbose = Bool(False, help='').tag(config=True)
    classes = List([PredictionConfig])

    config_file = Path('prediction_config.py', help='Load this config file')

    aliases = Dict({
        'log_level': 'Application.log_level',
        })

    flags = Dict({
        'verbose': ({
            'PredictionApplication': {
                'verbose': True,
                },
            }, 'set verbose mode'),
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

        self.load_config_file(
            self.prediction_config.load_dir
            / TrainingApplication().config_file.name)
        self.dataset_config = DatasetConfig(config=self.config)
        self.model_config = ModelConfig(config=self.config)

    def start(self):
        pc = self.prediction_config
        tag_xyz_map, pc.elements = parse_xyz(pc.data_file)
        datasets = self.construct_datasets(tag_xyz_map)
        datasets = DatasetGenerator(*datasets).all()
        results = self.predict(datasets)
        self.dump_result(results)
        shutil.copy(self.config_file, pc.load_dir / self.config_file.name)

    def construct_datasets(self, tag_xyz_map):
        dc = self.dataset_config
        pc = self.prediction_config
        if 'all' in pc.tags:
            included_tags = sorted(tag_xyz_map)
        else:
            included_tags = pc.tags

        preprocesses = []
        for (name, args, kwargs) in dc.preprocesses:
            preprocess = PREPROCESS[name](*args, **kwargs)
            preprocess.load(pc.load_dir / 'preprocess' / f'{name}.npz',
                            verbose=self.verbose)
            preprocesses.append(preprocess)

        datasets = []
        for tag in included_tags:
            try:
                tagged_xyz = tag_xyz_map[tag]
            except KeyError:
                pprint(f'Sub dataset tagged as "{tag}" does not exist.')
                continue

            pprint(f'Construct sub dataset tagged as "{tag}"')
            dataset = HDNNPDataset(dc.descriptor, dc.property_, order=pc.order)
            structures = [AtomicStructure(atoms) for atoms
                          in ase.io.iread(str(tagged_xyz),
                                          index=':', format='xyz')]

            dataset.descriptor.make(structures, verbose=self.verbose,
                                    **dc.parameters)

            dataset.construct(pc.elements, preprocesses,
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

            predictions = hdnnp.predict(
                chainer.dataset.concat_examples(dataset))
            result = {
                **{'tag': dataset.tag},
                **{property_: prediction.data for property_, prediction
                   in zip(dataset.property.properties, predictions)},
                }
            results.append(result)
        return results

    def dump_result(self, results):
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
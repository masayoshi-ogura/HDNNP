# coding: utf-8

import pathlib

from traitlets import (
    Dict, Float, Instance, Integer, List, TraitType, Tuple, Unicode,
    )
from traitlets.config import Configurable

from hdnnpy.preprocess.preprocess_base import PreprocessBase


class Path(TraitType):
    default_value = '.'
    info_text = 'a pathlib.Path instance'

    def validate(self, obj, value):
        if isinstance(value, pathlib.Path):
            return value.absolute()
        elif isinstance(value, str):
            return pathlib.Path(value).absolute()
        else:
            self.error(obj, value)


class DatasetConfig(Configurable):
    n_sample = Integer(0, help='')
    descriptor = Unicode(help='configuration is required').tag(config=True)
    parameters = Dict(List, help='configuration is required').tag(config=True)
    property_ = Unicode(help='configuration is required').tag(config=True)
    order = Integer(help='configuration is required').tag(config=True)
    preprocesses = List(Instance(PreprocessBase), help='').tag(config=True)


class ModelConfig(Configurable):
    layers = List(Tuple(Integer, Unicode),
                  help='configuration is required').tag(config=True)
    order = Integer(help='configuration is required').tag(config=True)
    loss_function_params = Dict(help='').tag(config=True)


class TrainingConfig(Configurable):
    elements = List(Unicode, help='')
    data_file = Path(help='configuration is required').tag(config=True)
    tags = List(Unicode, ('all',), help='').tag(config=True)
    out_dir = Path('output', help='').tag(config=True)
    train_test_ratio = Float(0.9, help='').tag(config=True)
    init_lr = Float(1.0e-3, help='').tag(config=True)
    final_lr = Float(1.0e-6, help='').tag(config=True)
    lr_decay = Float(1.0e-6, help='').tag(config=True)
    l1_norm = Float(0.0e-4, help='').tag(config=True)
    l2_norm = Float(0.0e-4, help='').tag(config=True)
    interval = Integer(10, help='').tag(config=True)
    patients = Integer(5, help='').tag(config=True)
    epoch = Integer(help='configuration is required').tag(config=True)
    batch_size = Integer(help='configuration is required').tag(config=True)
    metrics = Unicode(help='configuration is required').tag(config=True)


class PredictionConfig(Configurable):
    load_dir = Path('output', help='').tag(config=True)

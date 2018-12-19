# coding: utf-8

import pathlib

from traitlets import (
    Bool, CaselessStrEnum, Dict, Float,
    Integer, List, TraitType, Tuple, Unicode,
    )
import traitlets.config


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


class Configurable(traitlets.config.Configurable):
    def dump(self):
        dic = {key: value for key, value in self._trait_values.items()
               if key not in ['config', 'parent']}
        return dic


class DatasetConfig(Configurable):
    # not configurable
    n_sample = Integer(
        help='Total number of data counted internally')

    # configurable
    descriptor = CaselessStrEnum(
        ['symmetry_function'],
        default_value='symmetry_function',
        help='Name of descriptor dataset used for input of HDNNP'
        ).tag(config=True)
    parameters = Dict(
        trait=List,
        help='Parameters used for the specified descriptor dataset. '
             'Set as Dict{key: List[Tuple(parameters)]}. '
             'This will be passed to descriptor dataset as keyword arguments. '
             'ex.) {"type2": [(5.0, 0.01, 2.0)]}'
        ).tag(config=True)
    property_ = CaselessStrEnum(
        ['interatomic_potential'],
        default_value='interatomic_potential',
        help='Name of property dataset to be optimized by HDNNP'
        ).tag(config=True)
    preprocesses = List(
        trait=Tuple(
            CaselessStrEnum(['normalization', 'pca', 'standardization']),
            Tuple(),
            Dict()
            ),
        help='Preprocess to be applied for input of HDNNP (=descriptor). '
             'Set as List[Tuple(Str(name), Tuple(args), Dict{kwargs})]. '
             'Each preprocess instance will be initialized with '
             '(*args, **kwargs). '
             'ex.) [("pca", (20,), {})]'
        ).tag(config=True)
    remake = Bool(
        default_value=False,
        help='If the given data file and the loaded dataset are not '
             'compatible, automatically recalculate and overwrite it.'
        ).tag(config=True)


class ModelConfig(Configurable):
    # configurable
    layers = List(
        trait=Tuple(Integer, Unicode),
        help='Structure of a neural network constituting HDNNP. '
             'Set as List[Tuple(Int(# of nodes), Str(activation function))]. '
             'Activation function of the last layer must be "identity". '
        ).tag(config=True)


class TrainingConfig(Configurable):
    # not configurable
    elements = List(
        trait=Unicode,
        help='All elements contained in the dataset listed internally')

    # configurable
    data_file = Path(
        help='Path to a data file used for HDNNP training. '
             'Only .xyz file format is supported.'
        ).tag(config=True)
    tags = List(
        trait=Unicode,
        default_value=['all'],
        help='List of data tags used for HDNNP training. '
             'If you set "all", all data contained in the data file is used.'
        ).tag(config=True)
    out_dir = Path(
        default_value='output',
        help='Path to output directory. '
             'NOTE: Currently, all output files will be overwritten.'
        ).tag(config=True)
    order = Integer(
        help='Order of differentiation used for calculation '
             'of descriptor & property datasets and HDNNP training. '
             'ex.) 0: energy, 1: force, for interatomic potential'
        ).tag(config=True)
    train_test_ratio = Float(
        default_value=0.9,
        help='Ratio to use for training data. '
             'The rest are used for test data.'
        ).tag(config=True)
    # chainer training
    loss_function = Tuple(
        CaselessStrEnum(['zeroth_only', 'first_only', 'mix']),
        Dict(),
        help='Name of loss function and parameters of it. '
             'Set as Tuple(Str(name), Dict{parameters}). '
             'ex.) ("mix", {"mixing_beta": 0.5})'
        ).tag(config=True)
    init_lr = Float(
        default_value=1.0e-3,
        help='Initial learning rate'
        ).tag(config=True)
    final_lr = Float(
        default_value=1.0e-6,
        help='Lower limit of learning rate when it decays'
        ).tag(config=True)
    lr_decay = Float(
        help='Rate of exponential decay of learning rate'
        ).tag(config=True)
    l1_norm = Float(
        help='Coefficient for the weight decay in L1 regularization'
        ).tag(config=True)
    l2_norm = Float(
        help='Coefficient for the weight decay in L2 regularization'
        ).tag(config=True)
    interval = Integer(
        help='Length of interval of training epochs used for checking metrics'
             ' value'
        ).tag(config=True)
    patients = Integer(
        help='Counts to let `chainer.training.triggers.EarlyStoppingTrigger`'
             ' be patient'
        ).tag(config=True)
    epoch = Integer(
        help='Upper bound of the number of training loops'
        ).tag(config=True)
    batch_size = Integer(
        help='Number of data within each batch'
        ).tag(config=True)
    # chainer extension flags
    scatter_plot = Bool(
        False,
        help='Set chainer training extension `ScatterPlot` if this flag is set'
        ).tag(config=True)
    log_report = Bool(
        True,
        help='Set chainer training extension `LogReport` if this flag is set'
        ).tag(config=True)
    print_report = Bool(
        True,
        help='Set chainer training extension `PrintReport` if this flag is set'
        ).tag(config=True)
    plot_report = Bool(
        False,
        help='Set chainer training extension `PlotReport` if this flag is set'
        ).tag(config=True)


class PredictionConfig(Configurable):
    # not configurable
    elements = List(
        trait=Unicode,
        help='All elements contained in the dataset listed internally')

    # configurable
    data_file = Path(
        help='Path to a data file used for HDNNP prediction. '
             'Only .xyz file format is supported.'
        ).tag(config=True)
    tags = List(
        trait=Unicode,
        default_value=['all'],
        help='List of data tags used for HDNNP prediction. '
             'If you set "all", all data contained in the data file is used.'
        ).tag(config=True)
    load_dir = Path(
        default_value='output',
        help='Path to directory to load training output files'
        ).tag(config=True)
    order = Integer(
        help='Order of differentiation used for calculation '
             'of descriptor & property datasets and HDNNP prediction. '
             'ex.) 0: energy, 1: force, for interatomic potential'
        ).tag(config=True)
    dump_format = CaselessStrEnum(
        ['.npz'],
        default_value='.npz',
        help='File format to output HDNNP predition result'
        ).tag(config=True)

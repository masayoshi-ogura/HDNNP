# coding=utf-8

import fnmatch
import pathlib
import shutil

import chainer
import chainer.training.extensions as ext
from chainer.training.triggers import EarlyStoppingTrigger
import chainermn
from traitlets import (Bool, Dict, List, Unicode)
from traitlets.config import Application
import yaml

from hdnnpy.cli.configurables import (
    DatasetConfig, ModelConfig, Path, TrainingConfig,
    )
from hdnnpy.dataset import (AtomicStructure, DatasetGenerator, HDNNPDataset)
from hdnnpy.dataset.descriptor import DESCRIPTOR_DATASET
from hdnnpy.dataset.property import PROPERTY_DATASET
from hdnnpy.format import parse_xyz
from hdnnpy.model import (HighDimensionalNNP, MasterNNP)
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.training import (
    LOSS_FUNCTION, Manager, Updater, ScatterPlot, set_log_scale,
    )
from hdnnpy.utils import (MPI, mkdir, pprint, pyyaml_path_representer)


class TrainingApplication(Application):
    name = Unicode(u'hdnnpy train')
    description = 'Train a HDNNP to optimize given properties.'

    is_resume = Bool(
        False,
        help='Resume flag used internally.')
    resume_dir = Path(
        None,
        allow_none=True,
        help='This option can be set only by command line.')
    verbose = Bool(
        False,
        help='Set verbose mode'
        ).tag(config=True)

    classes = List([DatasetConfig, ModelConfig, TrainingConfig])

    config_file = Path(
        'training_config.py',
        help='Load this config file')

    aliases = Dict({
        'resume': 'TrainingApplication.resume_dir',
        'log_level': 'Application.log_level',
        })

    flags = Dict({
        'verbose': ({
            'TrainingApplication': {
                'verbose': True,
                },
            }, 'Set verbose mode'),
        'v': ({
            'TrainingApplication': {
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
        self.training_config = None

    def initialize(self, argv=None):
        # temporarily set `resume_dir` configurable
        self.__class__.resume_dir.tag(config=True)
        self.parse_command_line(argv)

        if self.resume_dir is not None:
            self.is_resume = True
            self.config_file = self.resume_dir.with_name(self.config_file.name)
        self.load_config_file(self.config_file)

        self.dataset_config = DatasetConfig(config=self.config)
        self.model_config = ModelConfig(config=self.config)
        self.training_config = TrainingConfig(config=self.config)
        if self.is_resume:
            self.training_config.out_dir = self.resume_dir.parent

    def start(self):
        tc = self.training_config
        mkdir(tc.out_dir)
        tag_xyz_map, tc.elements = parse_xyz(
            tc.data_file, verbose=self.verbose)
        datasets = self.construct_datasets(tag_xyz_map)
        dataset = DatasetGenerator(*datasets).holdout(tc.train_test_ratio)
        try:
            result = self.train(dataset)
        except InterruptedError as e:
            pprint(e)
        else:
            self.dump_result(result)
        finally:
            if not self.is_resume:
                shutil.copy(self.config_file,
                            tc.out_dir / self.config_file.name)

    def construct_datasets(self, tag_xyz_map):
        dc = self.dataset_config
        tc = self.training_config

        preprocess_dir = tc.out_dir / 'preprocess'
        mkdir(preprocess_dir)
        preprocesses = []
        for (name, args, kwargs) in dc.preprocesses:
            preprocess = PREPROCESS[name](*args, **kwargs)
            if self.is_resume:
                preprocess.load(
                    preprocess_dir / f'{name}.npz', verbose=self.verbose)
            preprocesses.append(preprocess)

        datasets = []
        for pattern in tc.tags:
            for tag in fnmatch.filter(tag_xyz_map, pattern):
                if self.verbose:
                    pprint(f'Construct sub dataset tagged as "{tag}"')
                tagged_xyz = tag_xyz_map.pop(tag)
                structures = AtomicStructure.read_xyz(tagged_xyz)

                # prepare descriptor dataset
                descriptor = DESCRIPTOR_DATASET[dc.descriptor](
                    tc.order, structures, **dc.parameters)
                descriptor_npz = tagged_xyz.with_name(f'{dc.descriptor}.npz')
                if descriptor_npz.exists():
                    descriptor.load(
                        descriptor_npz, verbose=self.verbose, remake=dc.remake)
                else:
                    descriptor.make(verbose=self.verbose)
                    descriptor.save(descriptor_npz, verbose=self.verbose)

                # prepare property dataset
                property_ = PROPERTY_DATASET[dc.property_](
                    tc.order, structures)
                property_npz = tagged_xyz.with_name(f'{dc.property_}.npz')
                if property_npz.exists():
                    property_.load(
                        property_npz, verbose=self.verbose, remake=dc.remake)
                else:
                    property_.make(verbose=self.verbose)
                    property_.save(property_npz, verbose=self.verbose)

                # construct HDNNP dataset from descriptor & property datasets
                dataset = HDNNPDataset(descriptor, property_)
                dataset.construct(
                    all_elements=tc.elements, preprocesses=preprocesses,
                    shuffle=True, verbose=self.verbose)
                dataset.scatter()
                datasets.append(dataset)
                dc.n_sample += dataset.total_size

        for preprocess in preprocesses:
            preprocess.save(
                preprocess_dir / f'{preprocess.name}.npz',
                verbose=self.verbose)

        return datasets

    def train(self, dataset, comm=None):
        mc = self.model_config
        tc = self.training_config
        if comm is None:
            comm = chainermn.create_communicator('naive', MPI.comm)
        result = {'training_time': 0.0, 'observation': []}

        # model and optimizer
        master_nnp = MasterNNP(tc.elements, mc.layers)
        master_opt = chainer.optimizers.Adam(tc.init_lr)
        master_opt = chainermn.create_multi_node_optimizer(master_opt, comm)
        master_opt.setup(master_nnp)
        master_opt.add_hook(chainer.optimizer_hooks.Lasso(tc.l1_norm))
        master_opt.add_hook(chainer.optimizer_hooks.WeightDecay(tc.l2_norm))

        for training, test in dataset:
            tag = training.tag
            properties = training.property.properties

            # iterators
            train_iter = chainer.iterators.SerialIterator(
                training, tc.batch_size // MPI.size, repeat=True, shuffle=True)
            test_iter = chainer.iterators.SerialIterator(
                test, tc.batch_size // MPI.size, repeat=False, shuffle=False)

            # model
            hdnnp = HighDimensionalNNP(
                training.elemental_composition, mc.layers, tc.order)
            hdnnp.sync_param_with(master_nnp)
            main_opt = chainer.Optimizer()
            main_opt = chainermn.create_multi_node_optimizer(main_opt, comm)
            main_opt.setup(hdnnp)

            # loss function
            name, kwargs = tc.loss_function
            loss_function, observation_keys = (
                LOSS_FUNCTION[name](hdnnp, properties, **kwargs))

            # triggers
            interval = (tc.interval, 'epoch')
            stop_trigger = EarlyStoppingTrigger(
                check_trigger=interval,
                monitor=f'val/main/{observation_keys[-1]}',
                patients=tc.patients, mode='min',
                verbose=self.verbose, max_trigger=(tc.epoch, 'epoch'))

            # updater and trainer
            updater = Updater(train_iter,
                              {'main': main_opt, 'master': master_opt},
                              loss_func=loss_function)
            out_dir = tc.out_dir / tag
            trainer = chainer.training.Trainer(updater, stop_trigger, out_dir)

            # extensions
            trainer.extend(ext.ExponentialShift('alpha', 1 - tc.lr_decay,
                                                target=tc.final_lr,
                                                optimizer=master_opt))
            evaluator = chainermn.create_multi_node_evaluator(
                ext.Evaluator(test_iter, hdnnp, eval_func=loss_function), comm)
            trainer.extend(evaluator, name='val')
            if tc.scatter_plot:
                trainer.extend(ScatterPlot(test, hdnnp, comm),
                               trigger=interval)
            if MPI.rank == 0:
                if tc.log_report:
                    trainer.extend(ext.LogReport(log_name='training.log'))
                if tc.print_report:
                    trainer.extend(ext.PrintReport(
                        ['epoch', 'iteration']
                        + [f'main/{key}' for key in observation_keys]
                        + [f'val/main/{key}' for key in observation_keys]))
                if tc.plot_report:
                    trainer.extend(ext.PlotReport(
                        [f'main/{observation_keys[-1]}',
                         f'val/main/{observation_keys[-1]}'],
                        x_key='epoch', postprocess=set_log_scale,
                        file_name='RMSE.png', marker=None))

            manager = Manager(tag, trainer, result, is_snapshot=True)
            if self.is_resume:
                manager.check_to_resume(self.resume_dir.name)
            if manager.allow_to_run:
                with manager:
                    trainer.run()

        chainer.serializers.save_npz(tc.out_dir / 'master_nnp.npz', master_nnp)

        return result

    def dump_result(self, result):
        if MPI.rank != 0:
            return

        yaml.add_representer(pathlib.PosixPath, pyyaml_path_representer)
        result_file = self.training_config.out_dir / 'training_result.yaml'
        with result_file.open('w') as f:
            yaml.dump({
                'dataset': self.dataset_config.dump(),
                'model': self.model_config.dump(),
                'training': self.training_config.dump(),
                }, f, default_flow_style=False)
            yaml.dump({
                'result': result,
                }, f, default_flow_style=False)


def generate_config_file():
    training_app = TrainingApplication()
    training_app.config_file.write_text(training_app.generate_config_file())

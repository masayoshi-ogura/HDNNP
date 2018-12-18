# coding=utf-8

import os
import pathlib
import shutil
import sys
import yaml

import ase.io
import chainer
import chainer.training.extensions as ext
from chainer.training.triggers import EarlyStoppingTrigger
import chainermn
from traitlets import (Bool, Dict, List, Unicode)
from traitlets.config import Application

from hdnnpy.cli.configurables import (
    DatasetConfig, ModelConfig, Path, TrainingConfig,
    )
from hdnnpy.dataset import (AtomicStructure, DatasetGenerator, HDNNPDataset)
from hdnnpy.format import parse_xyz
from hdnnpy.model import (HighDimensionalNNP, MasterNNP)
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.training import (
    LOSS_FUNCTION, Evaluator, Manager, Updater, ScatterPlot, set_log_scale,
    )
from hdnnpy.utils import (MPI, mkdir, pprint)


class TrainingApplication(Application):
    name = Unicode(u'HDNNP training application')

    is_resume = Bool(False, help='')
    resume_dir = Path(None, allow_none=True, help='')
    verbose = Bool(False, help='').tag(config=True)
    classes = List([DatasetConfig, ModelConfig, TrainingConfig])

    config_file = Path('config.py', help='Load this config file')

    aliases = Dict({
        'resume': 'TrainingApplication.resume_dir',
        'log_level': 'Application.log_level',
        })

    flags = Dict({
        'verbose': ({
            'TrainingApplication': {
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
        self.training_config = None

    def initialize(self, argv=None):
        if MPI.rank != 0:
            sys.stdout = pathlib.Path(os.devnull).open('w')
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

        # assertion
        assert self.dataset_config.order == self.model_config.order

    def start(self):
        tc = self.training_config
        mkdir(tc.out_dir)
        tag_xyz_map, tc.elements = parse_xyz(tc.data_file)
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
        if 'all' in tc.tags:
            included_tags = sorted(tag_xyz_map)
        else:
            included_tags = tc.tags

        preprocess_dir = tc.out_dir / 'preprocess'
        mkdir(preprocess_dir)
        preprocesses = []
        for (name, args, kwargs) in dc.preprocesses:
            preprocess = PREPROCESS[name](*args, **kwargs)
            if self.is_resume:
                preprocess.load(preprocess_dir / f'{name}.npz',
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
            dataset = HDNNPDataset(descriptor=dc.descriptor,
                                   property_=dc.property_,
                                   order=dc.order)
            structures = [AtomicStructure(atoms) for atoms
                          in ase.io.iread(str(tagged_xyz),
                                          index=':', format='xyz')]

            descriptor_npz = tagged_xyz.with_name(f'{dc.descriptor}.npz')
            if descriptor_npz.exists():
                dataset.descriptor.load(descriptor_npz, verbose=self.verbose)
            else:
                dataset.descriptor.make(structures, verbose=self.verbose,
                                        **dc.parameters)
                dataset.descriptor.save(descriptor_npz, verbose=self.verbose)

            property_npz = tagged_xyz.with_name(f'{dc.property_}.npz')
            if property_npz.exists():
                dataset.property.load(property_npz, verbose=self.verbose)
            else:
                dataset.property.make(structures, verbose=self.verbose)
                dataset.property.save(property_npz, verbose=self.verbose)

            dataset.construct(tc.elements, preprocesses,
                              shuffle=True, verbose=self.verbose)
            for preprocess in preprocesses:
                preprocess.save(preprocess_dir / f'{preprocess.name}.npz',
                                verbose=self.verbose)

            dataset.scatter()
            datasets.append(dataset)
            dc.n_sample += dataset.total_size
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
                training.elemental_composition, mc.layers, mc.order)
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
                Evaluator(test_iter, hdnnp, eval_func=loss_function), comm)
            trainer.extend(evaluator, name='val')
            if tc.scatter_plot:
                trainer.extend(ScatterPlot(test, hdnnp, mc.order, comm),
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
                manager.check_resume(self.resume_dir.name)
            if manager.allow_to_run():
                with manager:
                    trainer.run()

        chainer.serializers.save_npz(tc.out_dir / 'master_nnp.npz', master_nnp)

        return result

    def dump_result(self, result):
        def represent_path(dumper, instance):
            return dumper.represent_scalar('Path', f'{instance}')

        yaml.add_representer(pathlib.PosixPath, represent_path)

        if MPI.rank == 0:
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


main = TrainingApplication.launch_instance


def generate_config_file():
    training_app = TrainingApplication()
    pathlib.Path('config.py').write_text(training_app.generate_config_file())

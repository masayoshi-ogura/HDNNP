# coding=utf-8

import datetime
import socket
import textwrap

import chainer
from traitlets import (CaselessStrEnum, Dict, Unicode)
from traitlets.config import Application
import yaml

from hdnnpy import __version__
from hdnnpy.cli.configurables import (DatasetConfig, ModelConfig, Path)
from hdnnpy.model import MasterNNP
from hdnnpy.preprocess import PREPROCESS
from hdnnpy.utils import (pprint, pyyaml_path_constructor)


class ConversionApplication(Application):
    name = Unicode(u'hdnnpy convert')
    description = 'Convert output files of training to required format.'

    format = CaselessStrEnum(
        ['lammps'],
        default_value='lammps',
        help='Name of the destination format.',
        ).tag(config=True)
    load_dir = Path(
        default_value='output',
        help='Path to directory to load training output files.',
        ).tag(config=True)

    aliases = Dict({
        'format': 'ConversionApplication.format',
        'load_dir': 'ConversionApplication.load_dir',
        })

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.training_result = None
        self.dataset_config = None
        self.model_config = None

    def initialize(self, argv=None):
        self.parse_command_line(argv)

        yaml.add_constructor('Path', pyyaml_path_constructor)
        self.training_result = yaml.load(
            (self.load_dir / 'training_result.yaml').open())
        self.dataset_config = DatasetConfig(**self.training_result['dataset'])
        self.model_config = ModelConfig(**self.training_result['model'])

    def start(self):
        tr = self.training_result
        dc = self.dataset_config
        mc = self.model_config

        # load preprocesses
        preprocesses = []
        for (name, args, kwargs) in dc.preprocesses:
            preprocess = PREPROCESS[name](*args, **kwargs)
            preprocess.load(
                self.load_dir / 'preprocess' / f'{name}.npz', verbose=False)
            preprocesses.append(preprocess)
        # load master nnp
        master_nnp = MasterNNP(tr['training']['elements'], mc.layers)
        chainer.serializers.load_npz(
            self.load_dir / 'master_nnp.npz', master_nnp)

        if self.format == 'lammps':
            self.dump_for_lammps(preprocesses, master_nnp)

    def dump_for_lammps(self, preprocesses, master_nnp):
        dc = self.dataset_config
        potential_file = self.load_dir / 'lammps.nnp'
        with potential_file.open('w') as f:
            # information
            now = datetime.datetime.now()
            machine = socket.gethostname()
            pprint(f'''
            # Created by hdnnpy {__version__} ({now}).
            # All parameters are read from [{machine}] {self.load_dir}.
            # Ref: https://github.com/ogura-edu/HDNNP
            ''', stream=f)

            # descriptor
            pprint(f'''
            # {dc.descriptor} parameters
            {len(dc.parameters)}
            ''', stream=f)
            for name, params in dc.parameters.items():
                params_str = ('\n'+' '*16).join([' '.join(map(str, row))
                                                 for row in params])
                pprint(f'''
                {name} {len(params)}
                {params_str}
                ''', stream=f)

            # preprocess
            pprint(f'''
            # pre-processing parameters
            {len(preprocesses)}
            ''', stream=f)
            for preprocess in preprocesses:
                pprint(f'''
                {preprocess.name}

                {textwrap.indent(
                    textwrap.dedent(preprocess.dump_params()), ' '*16)}
                ''', stream=f)

            # model
            pprint(f'''
            # neural network parameters
            {len(master_nnp[0])}

            {textwrap.indent(
                textwrap.dedent(master_nnp.dump_params()), ' '*12)}
            ''', stream=f)

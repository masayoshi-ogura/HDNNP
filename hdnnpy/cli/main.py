# coding=utf-8

import os
from pathlib import Path
import sys

from traitlets import Unicode
from traitlets.config import Application

from hdnnpy.cli.predict import PredictionApplication
from hdnnpy.cli.train import TrainingApplication
from hdnnpy.utils import MPI


class HDNNPApplication(Application):
    name = Unicode(u'hdnnpy')

    classes = [PredictionApplication, TrainingApplication]

    subcommands = {
        'predict': (PredictionApplication, PredictionApplication.description),
        'train': (TrainingApplication, TrainingApplication.description),
        }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize(self, argv=None):
        if MPI.rank != 0:
            sys.stdout = Path(os.devnull).open('w')
        super().initialize(argv)

    def start(self):
        super().start()

# coding=utf-8

import os
from pathlib import Path
import sys

from traitlets import Unicode
from traitlets.config import Application

from hdnnpy.cli.conversion_application import ConversionApplication
from hdnnpy.cli.prediction_application import PredictionApplication
from hdnnpy.cli.training_application import TrainingApplication
from hdnnpy.utils import MPI


class HDNNPApplication(Application):
    name = Unicode(u'hdnnpy')

    classes = [
        ConversionApplication,
        PredictionApplication,
        TrainingApplication,
        ]

    subcommands = {
        'convert': (ConversionApplication, ConversionApplication.description),
        'predict': (PredictionApplication, PredictionApplication.description),
        'train': (TrainingApplication, TrainingApplication.description),
        }

    def initialize(self, argv=None):
        if MPI.rank != 0:
            sys.stdout = Path(os.devnull).open('w')
        assert sys.argv[1] in self.subcommands, \
            'Only `hdnnpy train` and `hdnnpy predict` `hdnnpy convert` are' \
            ' available.'
        super().initialize(argv)


main = HDNNPApplication.launch_instance

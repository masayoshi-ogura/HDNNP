How to use HDNNP
================

.. contents::
   :local:
   :depth: 2


Data generation
-----------------

| Usually, HDNNP is used to reduce cost by learning the result of
  DFT(Density Functional Theory) calculation that is high accuracy and high cost.
| Therefore, first step is to generate training dataset using DFT calculation such as ab-initio MD calculation.



Pre-processing
-----------------

| HDNNP training application supports only .xyz file format.
| We prepare a python script to convert the output file of VASP such as ``OUTCAR`` to .xyz format file,
  but in the same way you can convert the output of other DFT calculation program to .xyz format file.
| Inside this program, file format conversion is performed using `ASE`_ package.

.. _ASE: https://wiki.fysik.dtu.dk/ase/ase/io/io.html




Training
-----------------

Configuration
^^^^^^^^^^^^^^^^^

A default configuration file for training is located in ``examples/training_config.py``.

``training_config.py`` consists of some subclasses that inherits ``traitlets.config.Configurable``:

* c.Application.xxx
* c.TrainingApplication.xxx
* c.DatasetConfig.xxx
* c.ModelConfig.xxx
* c.TrainingConfig.xxx


Following configurations are required, and remaining configurations are optional.

* c.DatasetConfig.parameters
* c.ModelConfig.layers
* c.TrainingConfig.data_file
* c.TrainingConfig.batch_size
* c.TrainingConfig.epoch
* c.TrainingConfig.order
* c.TrainingConfig.loss_function
* c.TrainingConfig.interval
* c.TrainingConfig.patients

For details of each setting, see ``training_config.py``


Command line interface
^^^^^^^^^^^^^^^^^^^^^^

Execute the following command in the directory where ``training_config.py`` is located.

::

    $ hdnnpy train

.. note::

    | Currently, if output directory set by ``c.TrainingConfig.out_dir`` already exists, it overwrites the existing file in the directory.
    | If you want to avoid this, please change ``c.TrainingConfig.out_dir`` for each execution.





Prediction
-----------------

Configuration
^^^^^^^^^^^^^^^^^

A default configuration file for prediction is located in ``examples/prediction_config.py``.

``prediction_config.py`` consists of some subclasses that inherits ``traitlets.config.Configurable``:

* c.Application.xxx
* c.PredictionApplication.xxx
* c.PredictionConfig.xxx


Following configurations are required, and remaining configurations are optional.

* c.PredictionConfig.data_file
* c.PredictionConfig.order

For details of each setting, see ``prediction_config.py``


Command line interface
^^^^^^^^^^^^^^^^^^^^^^

Execute the following command in the directory where ``prediction_config.py`` is located.

::

    $ hdnnpy predict


Post-processing
-----------------

| It is possible to calculate MD simulation with LAMMPS using trained HDNNP.
| However, it is also under development.
| We welcome your comments and suggestions.

`HDNNP-LAMMPS interface program <https://github.com/ogura-edu/HDNNP-LAMMPS.git>`_


Command line interface
^^^^^^^^^^^^^^^^^^^^^^

Execute the following command.

::

    $ hdnnpy convert

| 2 command line options are available, and no config file is used in this command.
| To see details of these options, use

::

    $ hdnnpy convert -h

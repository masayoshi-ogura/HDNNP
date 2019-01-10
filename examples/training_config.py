# Configuration file for hdnnpy train.

#------------------------------------------------------------------------------
# Application(SingletonConfigurable) configuration
#------------------------------------------------------------------------------

## This is an application.

## The date format used by logging formatters for %(asctime)s
#c.Application.log_datefmt = '%Y-%m-%d %H:%M:%S'

## The Logging format template
#c.Application.log_format = '[%(name)s]%(highlevel)s %(message)s'

## Set the log level by value or name.
#c.Application.log_level = 30

#------------------------------------------------------------------------------
# TrainingApplication(Application) configuration
#------------------------------------------------------------------------------

## Set verbose mode
#c.TrainingApplication.verbose = False

#------------------------------------------------------------------------------
# DatasetConfig(Configurable) configuration
#------------------------------------------------------------------------------

## Name of descriptor dataset used for input of HDNNP
#c.DatasetConfig.descriptor = 'symmetry_function'

## Parameters used for the specified descriptor dataset. Set as Dict{key:
#  List[Tuple(parameters)]}. This will be passed to descriptor dataset as keyword
#  arguments. ex.) {"type2": [(5.0, 0.01, 2.0)]}
#c.DatasetConfig.parameters = {}

## Preprocess to be applied for input of HDNNP (=descriptor). Set as
#  List[Tuple(Str(name), Tuple(args), Dict{kwargs})]. Each preprocess instance
#  will be initialized with (*args, **kwargs). ex.) [("pca", (20,), {})]
#c.DatasetConfig.preprocesses = []

## Name of property dataset to be optimized by HDNNP
#c.DatasetConfig.property_ = 'interatomic_potential'

## If the given data file and the loaded dataset are not compatible,
#  automatically recalculate and overwrite it.
#c.DatasetConfig.remake = False

#------------------------------------------------------------------------------
# ModelConfig(Configurable) configuration
#------------------------------------------------------------------------------

## Structure of a neural network constituting HDNNP. Set as List[Tuple(Int(# of
#  nodes), Str(activation function))]. Activation function of the last layer must
#  be "identity".
#c.ModelConfig.layers = []

#------------------------------------------------------------------------------
# TrainingConfig(Configurable) configuration
#------------------------------------------------------------------------------

## Number of data within each batch
#c.TrainingConfig.batch_size = 0

## Path to a data file used for HDNNP training. Only .xyz file format is
#  supported.
#c.TrainingConfig.data_file = '.'

## Upper bound of the number of training loops
#c.TrainingConfig.epoch = 0

## Lower limit of learning rate when it decays
#c.TrainingConfig.final_lr = 1e-06

## Initial learning rate
#c.TrainingConfig.init_lr = 0.001

## Length of interval of training epochs used for checking metrics value
#c.TrainingConfig.interval = 0

## Coefficient for the weight decay in L1 regularization
#c.TrainingConfig.l1_norm = 0.0

## Coefficient for the weight decay in L2 regularization
#c.TrainingConfig.l2_norm = 0.0

## Set chainer training extension `LogReport` if this flag is set
#c.TrainingConfig.log_report = True

## Name of loss function and parameters of it. Set as Tuple(Str(name),
#  Dict{parameters}). ex.) ("mix", {"mixing_beta": 0.5})
#c.TrainingConfig.loss_function = ()

## Rate of exponential decay of learning rate
#c.TrainingConfig.lr_decay = 0.0

## Order of differentiation used for calculation of descriptor & property
#  datasets and HDNNP training. ex.) 0: energy, 1: force, for interatomic
#  potential
#c.TrainingConfig.order = 0

## Path to output directory. NOTE: Currently, all output files will be
#  overwritten.
#c.TrainingConfig.out_dir = 'output'

## Counts to let `chainer.training.triggers.EarlyStoppingTrigger` be patient
#c.TrainingConfig.patients = 0

## Set chainer training extension `PlotReport` if this flag is set
#c.TrainingConfig.plot_report = False

## Set chainer training extension `PrintReport` if this flag is set
#c.TrainingConfig.print_report = True

## Set chainer training extension `ScatterPlot` if this flag is set
#c.TrainingConfig.scatter_plot = False

## List of dataset tags. Use dataset for HDNNP training in this order. Pattern
#  matching is available.
#c.TrainingConfig.tags = ['*']

## Ratio to use for training data. The rest are used for test data.
#c.TrainingConfig.train_test_ratio = 0.9

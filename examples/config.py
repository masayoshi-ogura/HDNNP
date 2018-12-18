# Configuration file for HDNNP training application.

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

## This is an application.

## 
#c.TrainingApplication.verbose = False

#------------------------------------------------------------------------------
# DatasetConfig(Configurable) configuration
#------------------------------------------------------------------------------

## configuration is required
#c.DatasetConfig.descriptor = ''

## configuration is required
#c.DatasetConfig.order = 0

## configuration is required
#c.DatasetConfig.parameters = {}

## 
#c.DatasetConfig.preprocesses = []

## configuration is required
#c.DatasetConfig.property_ = ''

#------------------------------------------------------------------------------
# ModelConfig(Configurable) configuration
#------------------------------------------------------------------------------

## configuration is required
#c.ModelConfig.layers = []

## configuration is required
#c.ModelConfig.order = 0

#------------------------------------------------------------------------------
# TrainingConfig(Configurable) configuration
#------------------------------------------------------------------------------

## configuration is required
#c.TrainingConfig.batch_size = 0

## configuration is required
#c.TrainingConfig.data_file = '.'

## configuration is required
#c.TrainingConfig.epoch = 0

## 
#c.TrainingConfig.final_lr = 1e-06

## 
#c.TrainingConfig.init_lr = 0.001

## 
#c.TrainingConfig.interval = 10

## 
#c.TrainingConfig.l1_norm = 0.0

## 
#c.TrainingConfig.l2_norm = 0.0

## Set chainer extension `LogReport`
#c.TrainingConfig.log_report = False

## configuration is required
#c.TrainingConfig.loss_function = ()

## 
#c.TrainingConfig.lr_decay = 1e-06

## 
#c.TrainingConfig.out_dir = 'output'

## 
#c.TrainingConfig.patients = 5

## Set chainer extension `PlotReport`
#c.TrainingConfig.plot_report = False

## Set chainer extension `PrintReport`
#c.TrainingConfig.print_report = False

## Set chainer extension `ScatterPlot`
#c.TrainingConfig.scatter_plot = False

## 
#c.TrainingConfig.tags = ['all']

## 
#c.TrainingConfig.train_test_ratio = 0.9

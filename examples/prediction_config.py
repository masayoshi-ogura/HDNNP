# Configuration file for hdnnpy predict.

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
# PredictionApplication(Application) configuration
#------------------------------------------------------------------------------

## Set verbose mode
#c.PredictionApplication.verbose = False

#------------------------------------------------------------------------------
# PredictionConfig(Configurable) configuration
#------------------------------------------------------------------------------

## Path to a data file used for HDNNP prediction. Only .xyz file format is
#  supported.
#c.PredictionConfig.data_file = '.'

## File format to output HDNNP predition result
#c.PredictionConfig.dump_format = '.npz'

## Path to directory to load training output files
#c.PredictionConfig.load_dir = 'output'

## Order of differentiation used for calculation of descriptor & property
#  datasets and HDNNP prediction. ex.) 0: energy, 1: force, for interatomic
#  potential
#c.PredictionConfig.order = 0

## List of data tags used for HDNNP prediction. If you set "all", all data
#  contained in the data file is used.
#c.PredictionConfig.tags = ['all']

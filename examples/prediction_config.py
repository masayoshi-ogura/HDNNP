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

## 
#c.PredictionApplication.verbose = False

#------------------------------------------------------------------------------
# PredictionConfig(Configurable) configuration
#------------------------------------------------------------------------------

## configuration is required
#c.PredictionConfig.data_file = '.'

## 
#c.PredictionConfig.dump_format = '.npz'

## 
#c.PredictionConfig.load_dir = 'output'

## configuration is required
#c.PredictionConfig.order = 0

## 
#c.PredictionConfig.tags = ['all']

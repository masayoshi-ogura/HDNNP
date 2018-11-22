# -*- coding: utf-8 -*-
"""Class Logger for creating instances for logging easily.

This module implements `Logger` class for logging in other modules.
It uses builder pattern for implementing instance configuration.

Example:
    From another module import and just build for usage.    
    Atomic implementation is this.

    ```python
    from logger import Logger

    log = Logger().build()
    log.info("hello from info")
    ```

    And here is the ouput.

    ```shell
    INFO 2018-11-22 21:38:06,552 vasp2xyz in line 4: hello from info
    ```
.. _PEP 484:
        https://www.python.org/dev/peps/pep-0484/
"""
import logging
from logging import StreamHandler, Formatter

class Logger(object):
    DEFAULT_LOG_LEVEL = logging.INFO
    """int: default log level of the logger"""

    def __init__(self, name="default"):
        """Fucntion of initializer `__init__`

        Args:
            name (str, optional): string name of this logger
        """
        self._logger_name = name
        self._log_level = self.DEFAULT_LOG_LEVEL
        self._format = Formatter('%(levelname)s %(asctime)s %(module)s line %(lineno)d: %(message)s')
        self._handlers = [{
          'handler': StreamHandler(),
          'level': self.DEFAULT_LOG_LEVEL  
        }]
        return self

    def set_log_level(self, level):
        """Function to set log level
        
        set_log_level configures the log level of the logger.\n
        Note that the log level for every handler are configured same.

        Args:
            level (int): log level for every handler

        Returns:
            self (object): Object myself
        """

        self._log_level = level
        for h in self._handlers:
            h['level'] = level
        return self

    def set_format(self, format_str):
        """Function to set log format
        
        set_format configures the log format of the logger.\n
        
        Note:
            the log format for every handler are configured same.

        Args:
            format_str (str): log format in string

        Returns:
            self (object): Object myself
        """

        self._format = Formatter(format_str)
        return self
    
    def add_handler(self, handler):
        """Fuction to add handler of logger

        it adds a handler to the list of handler in this class

        Args:
            handler (dict): a dict of `handler` and log `level`

        Returns:
            self (:obj;`:obj`, `int`): Object myself
        """

        self._handlers.append(handler)
        return self

    def build(self):
        """Fuction that builds a instance of `logger`

        it adds a handler to the list of handler in this class

        Args:
            None

        Returns:
            log (obj): logger object in logging
        """

        log = logging.getLogger(self._logger_name)
        log.setLevel(self._log_level)

        for h in self._handlers:
            h['handler'].setLevel(h['level'])
            h['handler'].setFormatter(self._format)
            log.addHandler(h['handler'])

        return log
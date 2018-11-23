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

    @classmethod
    def set_default_log_level(cls, level):
        cls.DEFAULT_LOG_LEVEL = level

    def __init__(self, name="default"):
        """Fucntion of initializer `__init__`

        Args:
            name (str, optional): string name of this logger
        """
        self.__logger = None
        self.__logger_name = name
        self.__log_level = self.DEFAULT_LOG_LEVEL
        self.__format = Formatter('%(levelname)s %(asctime)s %(module)s in line %(lineno)d: %(message)s')
        self.__handlers = [{
          'handler': StreamHandler(),
          'level': self.DEFAULT_LOG_LEVEL  
        }]
    
    def get_logger_name(self):
        """Fuction for getter of log_name

        Returns:
            logger_name (str): name of the logger
        """
        return self.__logger_name

    def get_log_level(self):
        """Fuction for getter of log_level

        Returns:
            logger_level (int): level of the logger
        """
        return self.__log_level

    def set_log_level(self, level):
        """Function to set log level
        
        set_log_level configures the log level of the logger.\n
        Note that the log level for every handler are configured same.

        Args:
            level (int): log level for every handler

        Returns:
            self (object): Object myself
        """

        self.__log_level = level
        for h in self.__handlers:
            h['level'] = level
        return self

    def get_format(self):
        """Fuction for getter of format property

        Returns:
            format (obj): Formatter object
        """
        return self.__format

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

        self.__format = Formatter(format_str)
        return self
    
    def get_handlers(self):
        """Fuction for getter of handlers property

        Returns:
            handlers (:obj; `obj`, `int): Dictonary of handlers and its log level
        """
        return self.__handlers

    def add_handler(self, handler):
        """Fuction to add handler of logger

        it adds a handler to the list of handler in this class

        Args:
            handler (dict): a dict of `handler` and log `level`

        Returns:
            self: Object myself
        """

        self.__handlers.append(handler)
        return self

    def build(self):
        """Fuction that builds a instance of `logger`

        it adds a handler to the list of handler in this class

        Args:
            None

        Returns:
            log (obj): logger object in logging
        """

        log = logging.getLogger(self.__logger_name)
        log.setLevel(self.__log_level)

        for h in self.__handlers:
            h['handler'].setLevel(h['level'])
            h['handler'].setFormatter(self.__format)
            log.addHandler(h['handler'])

        return log
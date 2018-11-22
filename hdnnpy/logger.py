import logging
from logging import StreamHandler, Formatter


class Logger(object):
    def __init__(self, name="default"):
        self.logger_name = name
        self._log_level = logging.INFO
        self._format = Formatter('%(levelname)s %(asctime)s %(module)s: %(message)s')
        self._handlers = [{
          'handler': StreamHandler(),
          'level': logging.INFO  
        }]

    def set_level(self, level):
        self._log_level = level
        for h in self._handlers:
            h['level'] = level

    def set_format(self, format_str):
        self._format = Formatter(format_str)
    
    def add_handler(self, handler):
        self._handlers.append(handler)

    def build(self):
        log = logging.getLogger(self.logger_name)
        log.setLevel(self._log_level)

        for h in self._handlers:
            h['handler'].setLevel(h['level'])
            h['handler'].setFormatter(self._format)
            log.addHandler(h['handler'])

        return log
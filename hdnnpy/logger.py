import logging
from logging import StreamHandler, Formatter

class Logger(object):
    def __init__(self, name="default"):
        self.logger_name = name
        self._log_level = logging.INFO
        self._stream_log_level =  logging.INFO
        self._format = Formatter('%(levelname)s %(asctime)s %(module)s: %(message)s')

    def set_level(self, level):
        self._log_level = level
        self._stream_log_level = level

    def set_format(self, format_str):
        self._format = Formatter(format_str)

    def build(self):
        log = logging.getLogger(self.logger_name)
        log.setLevel(self._log_level)
        stream_handler = StreamHandler()
        stream_handler.setLevel(self._stream_log_level)
        stream_handler.setFormatter(self._format)
        log.addHandler(stream_handler)
        return log


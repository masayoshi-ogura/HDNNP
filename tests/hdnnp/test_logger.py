# -*- coding: utf-8 -*-
import logging
from unittest import TestCase
from hdnnpy.logger import Logger
from logging import Formatter

class TestLogger(TestCase):
    
    def setUp(self):
        self.log = Logger()

    def test_init(self):
        log = self.log.build()
        self.assertIsInstance(self.log, Logger, "init should return Logger object")
        self.assertEqual(self.log.get_logger_name(), "default")
        self.assertEqual(self.log.get_log_level(), logging.INFO)

    def test_set_log_level(self):
        log = self.log.set_log_level(logging.ERROR)
        self.assertEqual(log.get_log_level(), logging.ERROR, 40)
        self.assertEqual(log.get_handlers()[0]["level"], logging.ERROR, 40)

    def test_set_format(self):
        test_format_str = '%(message)s'
        log = self.log.set_format(test_format_str)
        self.assertIsInstance(log.get_format(), Formatter)

    def test_build(self):
        log = self.log.build()
        self.assertTrue(not isinstance(log, Logger))
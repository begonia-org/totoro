#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2024/09/06 16:52:19
@Desc    :
'''

from totoro.config import logger

nlp_logger = logger.bind(name="nlp")
task_logger = logger.bind(name="task")
test_logger = logger.bind(name="test")


def init():
    global nlp_logger
    global task_logger
    global test_logger
    nlp_logger = logger.bind(name="nlp")
    task_logger = logger.bind(name="task")
    test_logger = logger.bind(name="test")

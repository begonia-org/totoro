#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   logger.py
@Time    :   2024/09/06 16:52:19
@Desc    :
'''
import sys

from loguru import logger


logger.remove()

logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time}</green> <cyan>{file}</cyan>:<cyan>{line}</cyan> <level>{message}</level>",
    level="INFO")

nlp_logger = logger.bind(name="nlp")
task_logger = logger.bind(name="task")
test_logger = logger.bind(name="test")

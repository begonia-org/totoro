#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2024/11/04 11:16:29
@Desc    :   Configuration module with singleton pattern and customizable project root.
'''

import os
import sys
from dynaconf import Dynaconf
from typing import Optional
from loguru import logger


logger.remove()


class TotoroConfigure:
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls, project_root: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super(TotoroConfigure, cls).__new__(cls)

            # Determine the project root, allowing for customization
            if project_root is None:
                project_root = os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__)))

            # Initialize Dynaconf settings
            cls._instance.settings = Dynaconf(
                envvar_prefix="TOTORO",
                environments=True,
                root_path=project_root,
                settings_files=[
                    'settings.yaml',
                    '.secrets.yaml',
                    "settings.dev.yaml",
                    "settings.prod.yaml",
                    "settings.test.yaml"
                ],
            )

            # Set additional settings attributes
            cls._instance.settings["ROOT_PATH"] = project_root

        return cls._instance

    @property
    def project_root(self) -> str:
        return self.settings.ROOT_PATH

    @property
    def logger_level(self) -> str:
        return self.settings.log.level.upper()

    @property
    def models_downloads(self) -> str:
        return self.settings.models.download.dir

    @property
    def maximum_docs(self) -> int:
        return int(self.settings.doc.maximum) * 1024 * 1024

    @property
    def model_dir(self) -> str:
        return self.settings.models.download.dir

    @property
    def light(self) -> int:
        return getattr(self.settings.models, "light", int(os.environ.get('LIGHTEN', "0")))

    @property
    def totoro_dir(self) -> str:
        return os.path.dirname(__file__)

    @property
    def rdb_config(self):
        return self.settings.rdb


# Usage example:
# config = TotoroConfigure("/custom/project/root")
# project_root = config.project_root
# cfg = None


def init(project_root=None):
    # global cfg
    cfg = TotoroConfigure(project_root=project_root)
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> <cyan>{file}</cyan>:<cyan>{line}</cyan> <level>{message}</level>",
        level=cfg.logger_level)

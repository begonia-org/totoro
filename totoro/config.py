#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   config.py
@Time    :   2024/11/04 11:16:29
@Desc    :   
'''


import os
from dynaconf import Dynaconf

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
settings = Dynaconf(
    envvar_prefix="TOTORO",
    environments=True,
    root_path=project_root,
    settings_files=['settings.yaml', '.secrets.yaml',
                    "settings.dev.yaml", "settings.prod.yaml", "settings.test.yaml"],
)

# `envvar_prefix` = export envvars with `export DYNACONF_FOO=bar`.
# `settings_files` = Load these files in the order.


settings["ROOT_PATH"] = os.path.dirname(os.path.abspath(__file__))


class TotoroConfigure:

    @staticmethod
    def get_project_root() -> str:
        return settings.ROOT_PATH

    @staticmethod
    def get_logger_level() -> str:
        return settings.log.level.upper()

    @staticmethod
    def get_models_downloads() -> str:
        return settings.models.download.dir

    @staticmethod
    def get_maxinum_docs() -> int:
        return int(settings.doc.maximum) * 1024 * 1024

    @staticmethod
    def model_dir():
        return settings.models.download.dir

    @staticmethod
    def light():
        return getattr(settings.models, "light") or int(os.environ.get('LIGHTEN', "0"))

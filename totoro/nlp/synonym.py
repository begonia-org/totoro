#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   synonym.py
@Time    :   2024/11/01 09:53:53
@Desc    :   
'''


import json
import os
import time
import logging
import re

from totoro.config import TotoroConfigure as cfg


class Synonym:
    def __init__(self, redis=None):

        self.lookup_num = 100000000
        self.load_tm = time.time() - 1000000
        self.dictionary = None
        path = os.path.join(cfg.get_project_root(), "res", "synonym.json")
        try:
            with open(path, 'r') as json_file:
                self.dictionary = json.load(json_file)
        except Exception as e:
            logging.warn("Missing synonym.json")
            self.dictionary = {}

        if not redis:
            logging.warning(
                "Realtime synonym is disabled, since no redis connection.")
        if not len(self.dictionary.keys()):
            logging.warning(f"Fail to load synonym")

        self.redis = redis
        self.load()

    def load(self):
        if not self.redis:
            return

        if self.lookup_num < 100:
            return
        tm = time.time()
        if tm - self.load_tm < 3600:
            return

        self.load_tm = time.time()
        self.lookup_num = 0
        d = self.redis.get("kevin_synonyms")
        if not d:
            return
        try:
            d = json.loads(d)
            self.dictionary = d
        except Exception as e:
            logging.error("Fail to load synonym!" + str(e))

    def lookup(self, tk):
        self.lookup_num += 1
        self.load()
        res = self.dictionary.get(re.sub(r"[ \t]+", " ", tk.lower()), [])
        if isinstance(res, str):
            res = [res]
        return res


if __name__ == '__main__':
    dl = Synonym()
    print(dl.dictionary)

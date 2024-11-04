#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   chunker.py
@Time    :   2024/09/08 20:34:54
@Desc    :
'''

from abc import ABC, abstractmethod


class ChunkBuilder(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def chunk(self, filename, doc_name, binary=None, from_page=0, to_page=100000,
              lang="Chinese", callback=None, **kwargs) -> list:
        pass

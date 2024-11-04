#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   __.init__.py
@Time    :   2024/09/06 16:35:57
@Desc    :
'''
from typing import Dict

from totoro.models import constant_model

from .chunker import ChunkBuilder
from .naive import NaiveChunkBuilder

CHUNK_BUILDERS: Dict[constant_model.ChunkType, ChunkBuilder] = {
    constant_model.ChunkType.CHUNK_TYPE_NAIVE: NaiveChunkBuilder()
}

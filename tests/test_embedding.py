#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test_embedding.py
@Time    :   2024/11/04 10:29:29
@Desc    :   
'''

import os
import pytest

from totoro.biz.doc import DocEmbedding
from totoro.models.constant_model import ChunkType
from totoro.llm.embbeding.embedding import EmbeddingModel
from totoro.utils.logger import test_logger


@pytest.fixture(scope="class")
def setup_embedding(request):
    request.cls.embedding = DocEmbedding()
    request.cls.data_dir = os.path.join(os.path.dirname(__file__), "data")
    test_logger.debug(request.cls.data_dir)


@pytest.mark.usefixtures("setup_embedding")
class TestEmbedding:
    def test_docx(self):
        test_logger.debug("test")
        file = os.path.join(self.data_dir, "test_zh.docx")
        ret = self.embedding.build_embedding(
            file, "test_doc", ChunkType.CHUNK_TYPE_NAIVE, 1, EmbeddingModel["BAAI"]("", "BAAI/bge-m3"))
        for doc, tk in ret:
            test_logger.debug(doc, tk)

    def setup(self):
        self.embedding = DocEmbedding()
        self.data_dir = os.path.join(os.path.dirname(__file__), "data")
        test_logger.debug(self.data_dir)

    def teardown(self):
        pass

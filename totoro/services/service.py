#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   service.py
@Time    :   2024/11/04 10:03:09
@Desc    :   
'''
import tempfile
import shutil
import os
from typing import IO

import httpx
from totoro.biz.core import RAGCore
from totoro.pb import services_pb2, services_pb2_grpc, doc_pb2
from totoro.models.constant_model import ChunkType
from totoro.llm.embbeding.embedding import EmbeddingModel
from totoro.llm.rerank.rerank import RerankModel
from totoro.utils.utils import is_url


class RAGService(services_pb2_grpc.RAGCoreServiceServicer):
    def __init__(self):
        self.__core = RAGCore()

    def embedding(self, request: services_pb2.EmbeddingRequest, context) -> services_pb2.EmbeddingResponse:
        if not request.embedding or request.embedding.find("/") == -1:
            raise ValueError("No embedding model or invalid embedding model")
        if not request.chunk_type:
            raise ValueError("No chunk type")
        if not request.file_key_or_url:
            raise ValueError("No file key or url")
        factory = request.embedding.split("/")[0]
        with tempfile.TemporaryFile() as tmp:
            file = self.__get_file(tmp, request.file_key_or_url)
            if not isinstance(file, str):
                file = file.name
            doc_title = self.__get_doc_title(
                request.file_key_or_url, request.doc_title)
            ret = self.__core.build_embedding(
                file, doc_title, ChunkType.CHUNK_TYPE_NAIVE,
                1, EmbeddingModel[factory](request.model_api_key, request.embedding))
            items: doc_pb2.EmbededItem = []
            for doc, tk in ret:
                items.append(doc_pb2.EmbededItem(doc.to_protobuf(), tk))
            return services_pb2.EmbeddingResponse(items=items)

    def reanking(self, request: services_pb2.ReankingRequest, context):
        factory = request.rerank.split("/")[0]
        return self.__core.reranking(request.query, request.candidates,
                                   RerankModel[factory](
                                       request.model_api_key, request.rerank),
                                   request.keyword_simlarity_weight,
                                   request.semantic_simlarity_weight)

    def query(self, request: services_pb2.QueryBuildRequest, context) -> services_pb2.QueryBuildResponse:
        factory = request.embedding.split("/")[0]
        return self.__core.build_query_vector(request.query,
                                            EmbeddingModel[factory](request.model_api_key,
                                                                    request.embedding))

    def __get_file(self, tmp: IO, file_key_or_url: str):
        if is_url(file_key_or_url):
            with httpx.stream("GET", file_key_or_url) as r:
                for chunk in r.iter_bytes():
                    tmp.write(chunk)
            tmp.seek(0)
            return tmp
        shutil.copy2(file_key_or_url, tmp.name)
        return file_key_or_url

    def __get_doc_title(self, file_key_or_url: str, doc_title: str):
        return doc_title or os.path.basename(file_key_or_url)

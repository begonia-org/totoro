#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2024/11/04 14:23:27
@Desc    :   
'''

import os

import json
from typing import List
import faiss
import numpy as np

from totoro.biz.core import EmbeddingCore
from totoro.models.constant_model import ChunkType
from totoro.models.doc_model import Doc
from totoro.llm.embbeding.embedding import EmbeddingModel
from totoro.utils.logger import test_logger


def main():
    embedding = EmbeddingCore()
    data_dir = os.path.join(os.path.dirname(__file__), "tests", "data")
    test_logger.debug(data_dir)
    file = os.path.join(data_dir, "test_zh.docx")
    ret = embedding.build_embedding(
        file, "test_zh", ChunkType.CHUNK_TYPE_NAIVE, 1, EmbeddingModel["BAAI"]("", "BAAI/bge-large-zh-v1.5"))
    docs = []
    chunks: List[Doc] = []
    index = faiss.IndexFlatL2(1024)
    for doc, tk in ret:
        # print(doc, tk)
        chunks.append(doc)
        docs.append(doc.model_dump_json())
        faiss_vector = np.array(doc.q_vec, dtype=np.float32)
        index.add(faiss_vector.reshape(1, -1))
        # 转换为 np.float32 类型

        # test_logger.debug(doc, tk)
    with open("embedding_zh.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)
    file = os.path.join(data_dir, "中国国际航空航天博览会.pdf")
    ret = embedding.build_embedding(
        file, "中国国际航空航天博览会", ChunkType.CHUNK_TYPE_NAIVE, 1, EmbeddingModel["BAAI"]("", "BAAI/bge-large-zh-v1.5"))
    docs = []
    # index = faiss.IndexFlatL2(1024)
    for doc, _ in ret:
        # print(doc, tk)
        chunks.append(doc)
        docs.append(doc.model_dump_json())
        faiss_vector = np.array(doc.q_vec, dtype=np.float32)

        index.add(faiss_vector.reshape(1, -1))
        # 转换为 np.float32 类型

        # test_logger.debug(doc, tk)
    with open("embedding_zh_2.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=4)
    search_vec = embedding.build_query_vector(
        "科学技术普及法修订草案的目标是什么？", EmbeddingModel["BAAI"]("", "BAAI/bge-large-zh-v1.5"))
    # 将查询向量转换为 np.array，确保数据类型为 float32
    # 归一化查询向量
    query_vector = np.array([search_vec.query_vector], dtype=np.float32)
    # query_vector /= np.linalg.norm(query_vector, axis=1, keepdims=True)

    # # 在索引中查找 top_k 个相似向量
    distances, indices = index.search(query_vector, search_vec.top_k)
    filtered_results = [(idx, dist) for idx, dist in zip(
        indices[0], distances[0]) if dist >= 0.8 and dist <= 1.0]

    print("result:", filtered_results)


if __name__ == "__main__":
    main()

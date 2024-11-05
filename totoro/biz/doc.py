#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   doc.py
@Time    :   2024/11/01 15:57:58
@Desc    :   向量化文档
'''

import io
import re
from typing import List, Iterable, Tuple
import numpy as np
from totoro.models import constant_model, doc_model
from totoro.pb import doc_pb2

from totoro.chunker import CHUNK_BUILDERS
from totoro.utils.utils import rm_space
from totoro.models.doc_model import DocSearchVector
from ..llm.embbeding.embedding import BaseEmbedding


class DocEmbedding:
    def __init__(self):
        self.title_regx = re.compile(
            r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>")

    def chunk_file(
            self,
            chunk_type: constant_model.ChunkType,
            filename: str,
            doc_name: str,
            binary: io.BytesIO = None,
            from_page=0,
            to_page=100000,
            lang="Chinese",
            callback=None,
            **kwargs) -> List[doc_pb2.Doc]:
        chunker = CHUNK_BUILDERS.get(chunk_type)
        chunks = chunker.chunk(filename, doc_name, binary,
                               from_page, to_page, lang, callback, **kwargs)
        return chunks

    def embedding(self,
                  docs: List[doc_pb2.Doc],
                  embed: BaseEmbedding,
                  parser_config: doc_model.ParserConfig = doc_model.ParserConfig(
                      filename_embd_weight=0.1),
                  callback=None) -> Iterable[Tuple[doc_model.Doc, int]]:
        """
        embedding documents
        Args:
            docs (List[doc_pb2.Doc]): documents
            embed (BaseEmbedding): embedding model
            parser_config (doc_model.ParserConfig, optional): parser config. 
            Defaults to doc_model.ParserConfig(filename_embd_weight=0.1).
            callback ([type], optional): callback to monitor prog. Defaults to None.
        """
        # 设置每次批处理的大小为32
        batch_size = 32

        # 对每个文档的标题进行预处理，去掉空格。title_tokens 应该是文档的标题分词。
        tts, cnts = [rm_space(d.title_tokens) for d in docs if d.title_tokens], [
            self.title_regx.sub(" ", d.content_with_weight) for d in docs]  # 替换内容中的特定字符为一个空格

        # 初始化一个计数器，用于计算总的 token 数量
        tk_count = 0
        print(f"tts:{tts},cnts:{cnts}")
        # 如果标题和内容的数量相等，才继续处理
        if len(tts) == len(cnts):
            # 初始化一个空的 NumPy 数组，用于存储标题的向量表示
            tts_ = np.array([])

            # 批量处理标题
            for i in range(0, len(tts), batch_size):
                # 使用嵌入模型对标题文本进行编码，返回向量和处理的 token 数量
                vts, c = embed.encode(tts[i: i + batch_size])
                print(f"tts vts:{vts},c:{c},text:{tts[i: i + batch_size]}")

                # 如果这是第一个批次，则直接将结果赋值给 tts_
                if len(tts_) == 0:
                    tts_ = vts
                else:
                    # 否则将当前批次的结果与前面批次的结果进行拼接
                    tts_ = np.concatenate((tts_, vts), axis=0)

                # 更新 token 数量
                tk_count += c

                # 通过回调函数报告处理进度
                callback(prog=0.6 + 0.1 * (i + 1) /
                         len(tts), msg="encode title")

            # 将所有批次的标题向量合并
            tts = tts_

        # 初始化一个空的 NumPy 数组，用于存储内容的向量表示
        cnts_ = np.array([])

        # 批量处理内容
        for i in range(0, len(cnts), batch_size):
            # 使用嵌入模型对内容文本进行编码，返回向量和处理的 token 数量
            vts, c = embed.encode(cnts[i: i + batch_size])
            np.set_printoptions(precision=20)

            print(f"vts:{vts},c:{c}")
            # 如果这是第一个批次，则直接将结果赋值给 cnts_
            if len(cnts_) == 0:
                cnts_ = vts
            else:
                # 否则将当前批次的结果与前面批次的结果进行拼接
                cnts_ = np.concatenate((cnts_, vts), axis=0)

            # 更新 token 数量
            tk_count += c

            # 通过回调函数报告处理进度
            callback(prog=0.7 + 0.2 * (i + 1) /
                     len(cnts), msg="encode content")

        # 将所有批次的内容向量合并
        cnts = cnts_

        # 从 parser_config 中获取标题权重，并计算最终的向量
        title_w = float(parser_config.filename_embd_weight)
        print(f"title_w:{title_w},tts:{tts},cnts:{cnts}")
        # 如果标题和内容的长度相等，则按照权重将标题和内容的向量合并；否则，仅使用内容向量
        vects = (title_w * tts + (1 - title_w) *
                 cnts) if len(tts) == len(cnts) else cnts

        # 确保生成的向量与输入的文档数量一致
        assert len(vects) == len(docs)

        # 对每个文档，生成最终的向量并附加到文档中
        for i, d in enumerate(docs):
            v = vects[i].tolist()  # 将向量转为列表形式
            print(f"q_{len(v)}_vec: {v}")

            d.q_vec.extend(v)  # 将向量扩展到文档的 q_vec 字段中
            d.q_vec_size = len(v)  # 记录向量的大小

            # 如果文档没有 positions、top 或 page_num，则创建空列表
            if not d.positions:
                getattr(d, "positions").extend([])
                getattr(d, "top").extend([])
                getattr(d, "page_num").extend([])

            # 使用 yield 返回处理后的文档对象，并附带 token 数量
            yield doc_model.Doc.from_protobuf(d), tk_count

    def build_embedding(
            self,
            filename: str,
            doc_name: str,
            chunk_type: constant_model.ChunkType,
            tid: str,
            embed: BaseEmbedding,
            parser_config: doc_model.ParserConfig = doc_model.ParserConfig(
                filename_embd_weight=0.1, task_page_size=12),
            lang="Chinese",
    ) -> Iterable[Tuple[doc_model.Doc, int]]:
        """
        build embedding for a document

        Args:
            filename (str): file path
            doc_name (str): doc name
            chunk_type (constant_model.ChunkType): chunk type, e.g. constant_model.ChunkType.CHUNK_TYPE_NAIVE
            tid (str): task id
            embed (BaseEmbedding): embedding model
            parser_config (doc_model.ParserConfig, optional): parser config. 
            Defaults to doc_model.ParserConfig(filename_embd_weight=0.1, task_page_size=12).
            lang (str, optional): language. Defaults to "Chinese".
        Returns:
            Iterable[Tuple[doc_model.Doc, int]]: doc and token count
        """

        chunks = self.chunk_file(chunk_type, filename, doc_name,
                                 lang=lang, callback=self.callback(tid),
                                 parser_config=parser_config or doc_model.ParserConfig(
                                     filename_embd_weight=0.1, task_page_size=12))
        return self.embedding(chunks, embed, parser_config or doc_model.ParserConfig(
            filename_embd_weight=0.1, task_page_size=12), self.callback(tid))

    def build_query_vector(self, txt, emb_mdl: BaseEmbedding, sim=0.8, topk=10) -> DocSearchVector:
        qv, c = emb_mdl.encode_queries(txt)
        vec = {
            "q_vec_size": len(qv),
            "top_k": topk,
            "similarity": sim,
            "num_candidates": topk * 2,
            "query_vector": [float(v) for v in qv]
        }
        return DocSearchVector(**vec)

    def callback(self, tid):
        def call(prog=0.0, msg="ok"):
            pass
            # document_logger.debug(f"tid:{tid},progress:{prog},msg:{msg}")
            # self.task_repo.add_task_progress(
            #     tid, doc_model.DocDegreeProgress(message=msg, progress=prog))
            # self.rdb.set(cfg.get_doc_task_key(fid, kib),
            #              doc_model.DocDegreeProgress(message=msg, progress=prog).model_dump_json(),
            #              ex=cfg.get_doc_task_key_expire())

        return call

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
        batch_size = 32
        tts, cnts = [rm_space(d.title_tokens) for d in docs if d.title_tokens], [
            self.title_regx.sub(" ", d.content_with_weight) for d in docs]
        tk_count = 0
        if len(tts) == len(cnts):
            tts_ = np.array([])
            for i in range(0, len(tts), batch_size):
                vts, c = embed.encode(tts[i: i + batch_size])
                if len(tts_) == 0:
                    tts_ = vts
                else:
                    tts_ = np.concatenate((tts_, vts), axis=0)
                tk_count += c
                callback(prog=0.6 + 0.1 * (i + 1) /
                         len(tts), msg="encode title")
            tts = tts_

        cnts_ = np.array([])
        for i in range(0, len(cnts), batch_size):
            vts, c = embed.encode(cnts[i: i + batch_size])
            if len(cnts_) == 0:
                cnts_ = vts
            else:
                cnts_ = np.concatenate((cnts_, vts), axis=0)
            tk_count += c
            callback(prog=0.7 + 0.2 * (i + 1) /
                     len(cnts), msg="encode content")
        cnts = cnts_
        # print(f"parser_config:{parser_config}")
        title_w = float(parser_config.filename_embd_weight)
        # print(f"title w:{title_w}")
        vects = (title_w * tts + (1 - title_w) *
                 cnts) if len(tts) == len(cnts) else cnts

        assert len(vects) == len(docs)
        for i, d in enumerate(docs):
            v = vects[i].tolist()
            # getattr(d, f"q_{len(v)}_vec").extend(v)
            d.q_vec.extend(v)
            d.q_vec_size = len(v)
            # d.model_dump()
            if not d.positions:

                getattr(d, "positions").extend([])
                getattr(d, "top").extend([])
                getattr(d, "page_num").extend([])

            # print("top,", d.top)
            # print("page_num,", d.page_num)
            # print("positions,", d.positions)
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

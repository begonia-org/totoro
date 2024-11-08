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
from sklearn.metrics.pairwise import cosine_similarity as CosineSimilarity
from totoro.models import constant_model, doc_model
from totoro.pb import doc_pb2

from totoro.chunker import CHUNK_BUILDERS
from totoro.utils.utils import rm_space, rm_WWW, is_chinese, sub_special_char
from totoro.models.doc_model import DocSearchVector
from totoro.nlp.tokenizer import DocTokenizer
from totoro.nlp import term_weight, synonym
from ..llm.rerank.rerank import BaseRerank
from ..llm.embbeding.embedding import BaseEmbedding


class EmbeddingCore:
    def __init__(self):
        self.title_regx = re.compile(
            r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>")
        self.tokenizer = DocTokenizer()
        self.term_weight = term_weight.TermWeight()
        self.syn = synonym.Synonym()

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

    def rerank_by_model(self, rerank_mdl: BaseRerank, query: str,
                        candidate_tokens: List[str], tkweight=0.3,
                        vtweight=0.7) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        重排算法
        先通过 question_keywords 提取关键词，再利用 token_similarity 计算关键词相似度，同时调用预训练模型 rerank_mdl 计算语义相似度。
        最终通过权重合并得到综合相似度分数，用于对候选文本进行排序
        Args:
            rerank_mdl (BaseRerank): 预训练的重排模型
            query (str): 查询语句，使用自然语言描述的问题
            candidate_tokens (List[str]): candidate tokens
            tkweight (float, optional): 关键词相似度权重. Defaults to 0.3.
            vtweight (float, optional): 语义相似度权重. Defaults to 0.7.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 综合相似度分数，关键词相似度，语义相似度

        """
        keywords = self.question_keywords(query)

        tksim = self.token_similarity(keywords, candidate_tokens)
        vtsim, _ = rerank_mdl.similarity(
            query, [rm_space(" ".join(tks)) for tks in candidate_tokens])

        return tkweight*np.array(tksim) + vtweight*vtsim, tksim, vtsim

    def question_keywords(self, txt):
        """
        提取问题关键词
        Args:
            txt (str): 自然语言查询语句
        Returns:
        """
        txt = re.sub(
            r"[ :\r\n\t,，。？?/`!！&\^%%]+",
            " ",
            self.tokenizer.tradi2simp(
                self.tokenizer.strQ2B(
                    txt.lower()))).strip()
        txt = rm_WWW(txt)

        if not is_chinese(txt):
            return list(set([t for t in txt.split(" ") if t]))

        def need_fine_grained_tokenize(tk):
            if len(tk) < 3:
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):
                return False
            return True

        keywords = []
        for tt in self.term_weight.split(txt)[:256]:  # .split(" "):
            if not tt:
                continue
            keywords.append(tt)
            twts = self.term_weight.weights([tt])
            syns = self.syn.lookup(tt)
            if syns:
                keywords.extend(syns)
            # logging.info(json.dumps(twts, ensure_ascii=False))
            for tk, _ in sorted(twts, key=lambda x: x[1] * -1):
                sm = self.tokenizer.fine_grained_tokenize(tk).split(
                    " ") if need_fine_grained_tokenize(tk) else []
                sm = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "",
                        m) for m in sm]
                sm = [sub_special_char(m) for m in sm if len(m) > 1]
                sm = [m for m in sm if len(m) > 1]

                keywords.append(re.sub(r"[ \\\"']+", "", tk))
                keywords.extend(sm)
                if len(keywords) >= 12:
                    break

        return list(set(keywords))

    def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3,
                          vtweight=0.7):

        sims = CosineSimilarity([avec], bvecs)
        tksim = self.token_similarity(atks, btkss)
        return np.array(sims[0]) * vtweight + \
            np.array(tksim) * tkweight, tksim, sims[0]

    def token_similarity(self, atks, btkss):
        def toDict(tks):
            d = {}
            if isinstance(tks, str):
                tks = tks.split(" ")
            for t, c in self.term_weight.weights(tks, preprocess=False):
                if t not in d:
                    d[t] = 0
                d[t] += c
            return d

        atks = toDict(atks)
        btkss = [toDict(tks) for tks in btkss]
        return [self.similarity(atks, btks) for btks in btkss]

    def similarity(self, qtwt, dtwt):
        if isinstance(dtwt, type("")):
            dtwt = {t: w for t, w in self.term_weight.weights(
                self.term_weight.split(dtwt), preprocess=False)}
        if isinstance(qtwt, type("")):
            qtwt = {t: w for t, w in self.term_weight.weights(
                self.term_weight.split(qtwt), preprocess=False)}
        s = 1e-9
        for k, v in qtwt.items():
            if k in dtwt:
                s += v  # * dtwt[k]
        q = 1e-9
        for k, v in qtwt.items():
            q += v
        return s / q

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

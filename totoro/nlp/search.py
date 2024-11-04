#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   search.py
@Time    :   2024/11/01 09:54:06
@Desc    :   
'''


import json
import re
from copy import deepcopy

from elasticsearch_dsl import Q, Search
from typing import List, Optional, Dict, Tuple, Union
from dataclasses import dataclass

from totoro.nlp import tokenizer
from totoro.utils.logger import chat_logger
from totoro.utils.utils import rm_space
from .query import QueryBuilder
from totoro.biz.llm import LLMBundle

import numpy as np


def index_name(uid): return f"openrag_{uid}"


class SearchBuilder:
    def __init__(self, es):
        self.builder = QueryBuilder(es)
        self.builder.flds = [
            "title_tokens^10",
            "title_small_tokens^5",
            "important_keyword^30",
            "important_tokens^20",
            "content_tokens^2",
            "content_small_tokens"]
        # self.es = es

    @dataclass
    class SearchResult:
        total: int
        ids: List[str]
        query_vector: List[float] = None
        field: Optional[Dict] = None
        highlight: Optional[Dict] = None
        aggregation: Union[List, Dict, None] = None
        keywords: Optional[List[str]] = None
        group_docs: List[List] = None

    def _vector(self, txt, emb_mdl: LLMBundle, sim=0.8, topk=10):
        qv, c = emb_mdl.encode_queries(txt)
        return {
            "field": "q_%d_vec" % len(qv),
            "k": topk,
            "similarity": sim,
            "num_candidates": topk * 2,
            "query_vector": [float(v) for v in qv]
        }

    def search(self, req, index, emb_mdl=None) -> Tuple[Search, List[str]]:
        qst = req.get("question", "")
        bqry: Q = None
        bqry, keywords = self.builder.question(qst)

        def add_filters(bqry: Q):
            nonlocal req
            if req.get("kb_ids"):
                bqry.filter.append(Q("terms", kb_id=req["kb_ids"]))
            if req.get("doc_ids"):
                bqry.filter.append(Q("terms", doc_id=req["doc_ids"]))
            if "available" in req:
                if req["available"] == 0:
                    bqry.filter.append(Q("range", available={"lt": 1}))
                else:
                    bqry.filter.append(
                        Q("bool", must_not=Q("range", available={"lt": 1})))
            return bqry

        bqry = add_filters(bqry)
        bqry.boost = 0.05

        s = Search()
        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        src = req.get("fields",
                      ["doc_name_keyword",
                       "content_tokens",
                       "kb_id",
                       "img_id",
                       "title_tokens",
                       "important_keyword",
                       "image_id",
                       "doc_id",
                       "q_vec_size",
                       "q_512_vec",
                       "q_vec",
                       "q_768_vec",
                       "position_int",
                       "q_1024_vec",
                       "q_1536_vec",
                       "available",
                       "content_with_weight"])

        s = s.query(bqry)[pg * ps:(pg + 1) * ps]
        s = s.highlight("content_tokens")
        s = s.highlight("title_tokens")
        if not qst:
            if not req.get("sort"):
                s = s.sort(
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )
            else:
                s = s.sort(
                    {"page_num": {"order": "asc", "unmapped_type": "float",
                                  "mode": "avg", "numeric_type": "double"}},
                    {"top": {"order": "asc", "unmapped_type": "float",
                             "mode": "avg", "numeric_type": "double"}},
                    {"create_time": {"order": "desc", "unmapped_type": "date"}},
                    {"create_timestamp_flt": {
                        "order": "desc", "unmapped_type": "float"}}
                )

        if qst:
            s = s.highlight_options(
                fragment_size=120,
                number_of_fragments=5,
                boundary_scanner_locale="zh-CN",
                boundary_scanner="SENTENCE",
                boundary_chars=",./;:\\!()，。？：！……（）——、"
            )
        s = s.to_dict()
        q_vec = []
        if req.get("vector"):
            assert emb_mdl, "No embedding model selected"
            s["knn"] = self._vector(
                qst, emb_mdl, req.get(
                    "similarity", 0.1), topk)
            s["knn"]["filter"] = bqry.to_dict()
            if "highlight" in s:
                del s["highlight"]
            q_vec = s["knn"]["query_vector"]
        # chat_logger.debug("【Q】: {}".format(json.dumps(s)))
        # res = self.es.search(deepcopy(s), idxnm=idxnm, timeout="600s", src=src)
        return deepcopy(s), keywords
        # chat_logger.debug("TOTAL: {}".format(self.es.getTotal(res)))
        # if self.es.getTotal(res) == 0 and "knn" in s:
        #     bqry, _ = self.builder.question(qst, min_match="10%")
        #     bqry = add_filters(bqry)
        #     s["query"] = bqry.to_dict()
        #     s["knn"]["filter"] = bqry.to_dict()
        #     s["knn"]["similarity"] = 0.17
        #     res = self.es.search(s, idxnm=idxnm, timeout="600s", src=src)
        #     chat_logger.info("【Q】: {}".format(json.dumps(s)))

        # kwds = set([])
        # for k in keywords:
        #     kwds.add(k)
        #     for kk in tokenizer.fine_grained_tokenize(k).split(" "):
        #         if len(kk) < 2:
        #             continue
        #         if kk in kwds:
        #             continue
        #         kwds.add(kk)

        # aggs = self.getAggregation(res, "doc_name_keyword")

        # return self.SearchResult(
        #     total=self.es.getTotal(res),
        #     ids=self.es.getDocIds(res),
        #     query_vector=q_vec,
        #     aggregation=aggs,
        #     highlight=self.getHighlight(res),
        #     field=self.getFields(res, src),
        #     keywords=list(kwds)
        # )

    def getAggregation(self, res, g):
        if "aggregations" not in res or "aggs_" + g not in res["aggregations"]:
            return
        bkts = res["aggregations"]["aggs_" + g]["buckets"]
        return [(b["key"], b["doc_count"]) for b in bkts]

    def getHighlight(self, res):
        def rmspace(line):
            eng = set(list("qwertyuioplkjhgfdsazxcvbnm"))
            r = []
            for t in line.split(" "):
                if not t:
                    continue
                if len(r) > 0 and len(
                        t) > 0 and r[-1][-1] in eng and t[0] in eng:
                    r.append(" ")
                r.append(t)
            r = "".join(r)
            return r

        ans = {}
        for d in res["hits"]["hits"]:
            hlts = d.get("highlight")
            if not hlts:
                continue
            ans[d["_id"]] = "".join([a for a in list(hlts.items())[0][1]])
        return ans

    def getFields(self, sres, flds):
        res = {}
        if not flds:
            return {}
        for d in self.es.getSource(sres):
            m = {n: d.get(n) for n in flds if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, type([])):
                    m[n] = "\t".join([str(vv) if not isinstance(
                        vv, list) else "\t".join([str(vvv) for vvv in vv]) for vv in v])
                    continue
                if not isinstance(v, type("")):
                    m[n] = str(m[n])
                if n.find("tks") > 0:
                    m[n] = rm_space(m[n])

            if m:
                res[d["id"]] = m
        return res

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        chat_logger.info("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [tokenizer.tokenize(self.builder.remove_redundant_words(ck)).split(" ")
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.builder.hybrid_similarity(ans_v[i],
                                                                   chunk_v,
                                                                   tokenizer.tokenize(
                    self.qryr.rmWWW(pieces_[i])).split(" "),
                    chunks_tks,
                    tkweight, vtweight)
                mx = np.max(sim) * 0.99
                chat_logger.info("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_tokens"):
        _, keywords = self.qryr.question(query)
        ins_embd = [Dealer.trans2floats(sres.field[i].get("q_%d_vec" % len(
            sres.query_vector), "\t".join(["0"] * len(sres.query_vector)))) for i in sres.ids]
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_keyword", []), str):
                sres.field[i]["important_keyword"] = [
                    sres.field[i]["important_keyword"]]
        ins_tw = []
        for i in sres.ids:
            content_tokens = sres.field[i][cfield].split(" ")
            title_tokens = [t for t in sres.field[i].get(
                "title_tokens", "").split(" ") if t]
            important_keyword = sres.field[i].get("important_keyword", [])
            tks = content_tokens + title_tokens + important_keyword
            ins_tw.append(tks)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)
        return sim, tksim, vtsim

    def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_tokens"):
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_keyword", []), str):
                sres.field[i]["important_keyword"] = [
                    sres.field[i]["important_keyword"]]
        ins_tw = []
        for i in sres.ids:
            content_tokens = sres.field[i][cfield].split(" ")
            title_tokens = [t for t in sres.field[i].get(
                "title_tokens", "").split(" ") if t]
            important_keyword = sres.field[i].get("important_keyword", [])
            tks = content_tokens + title_tokens + important_keyword
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = rerank_mdl.similarity(
            " ".join(keywords), [rm_space(" ".join(tks)) for tks in ins_tw])

        return tkweight * np.array(tksim) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           tokenizer.tokenize(ans).split(" "),
                                           tokenizer.tokenize(inst).split(" "))

    def retrieval(self, question, embd_mdl, tenant_id, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True, rerank_mdl=None):
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "size": page_size,
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available": 1}
        search, keywords = self.search(req, index_name(tenant_id), embd_mdl)

        if rerank_mdl:
            sim, tsim, vsim = self.rerank_by_model(
                rerank_mdl, sres, question, 1 - vector_similarity_weight, vector_similarity_weight)
        else:
            sim, tsim, vsim = self.rerank(
                sres, question, 1 - vector_similarity_weight, vector_similarity_weight)
        idx = np.argsort(sim * -1)

        dim = len(sres.query_vector)
        start_idx = (page - 1) * page_size
        for i in idx:
            if sim[i] < similarity_threshold:
                break
            ranks["total"] += 1
            start_idx -= 1
            if start_idx >= 0:
                continue
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    continue
                break
            id = sres.ids[i]
            dnm = sres.field[id]["doc_name_keyword"]
            did = sres.field[id]["doc_id"]
            d = {
                "chunk_id": id,
                "content_tokens": sres.field[id]["content_tokens"],
                "content_with_weight": sres.field[id]["content_with_weight"],
                "doc_id": sres.field[id]["doc_id"],
                "doc_name_keyword": dnm,
                "kb_id": sres.field[id]["kb_id"],
                "important_keyword": sres.field[id].get("important_keyword", []),
                "img_id": sres.field[id].get("img_id", ""),
                "similarity": sim[i],
                "vector_similarity": vsim[i],
                "term_similarity": tsim[i],
                "vector": self.trans2floats(sres.field[id].get("q_%d_vec" % dim, "\t".join(["0"] * dim))),
                "positions": sres.field[id].get("position_int", "").split("\t")
            }
            if len(d["positions"]) % 5 == 0:
                poss = []
                for i in range(0, len(d["positions"]), 5):
                    poss.append([float(d["positions"][i]), float(d["positions"][i + 1]), float(d["positions"][i + 2]),
                                 float(d["positions"][i + 3]), float(d["positions"][i + 4])])
                d["positions"] = poss
            ranks["chunks"].append(d)
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                             v in sorted(ranks["doc_aggs"].items(),
                                         key=lambda x: x[1]["count"] * -1)]

        return ranks

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        sql = re.sub(r"[ `]+", " ", sql)
        sql = sql.replace("%", "")
        chat_logger.info(f"Get es sql: {sql}")
        replaces = []
        for r in re.finditer(r" ([a-z_]+_l?tks)( like | ?= ?)'([^']+)'", sql):
            fld, v = r.group(1), r.group(3)
            match = " MATCH({}, '{}', 'operator=OR;minimum_should_match=30%') ".format(
                fld, tokenizer.fine_grained_tokenize(tokenizer.tokenize(v)))
            replaces.append(
                ("{}{}'{}'".format(
                    r.group(1),
                    r.group(2),
                    r.group(3)),
                    match))

        for p, r in replaces:
            sql = sql.replace(p, r, 1)
        chat_logger.info(f"To es: {sql}")

        try:
            tbl = self.es.sql(sql, fetch_size, format)
            return tbl
        except Exception as e:
            chat_logger.error(f"SQL failure: {sql} =>" + str(e))
            return {"error": str(e)}

    def chunk_list(
        self,
        doc_id,
        tenant_id,
        max_count=1024,
        fields=[
            "doc_name_keyword",
            "content_with_weight",
            "img_id"]):
        s = Search()
        s = s.query(Q("match", doc_id=doc_id))[0:max_count]
        s = s.to_dict()
        es_res = self.es.search(s, idxnm=index_name(
            tenant_id), timeout="600s", src=fields)
        res = []
        for index, chunk in enumerate(es_res['hits']['hits']):
            res.append({fld: chunk['_source'].get(fld) for fld in fields})
        return res

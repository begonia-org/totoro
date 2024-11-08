#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   rerank.py
@Time    :   2024/11/07 10:50:36
@Desc    :   
'''


import json
import os
import re
import threading
from abc import ABC
from urllib.parse import urljoin

import numpy as np
import requests
import voyageai
import dashscope

from cohere import Client as CoClient

from huggingface_hub import snapshot_download
from qianfan.resources import Reranker

from totoro.config import TotoroConfigure as cfg
from totoro.utils.utils import num_tokens_from_string, truncate


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BaseRerank(ABC):
    def __init__(self, key, model_name, models_dir=cfg.model_dir()):
        self.models_dir = models_dir

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("Please implement encode method!")


class DefaultRerank(BaseRerank):
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key, model_name, **kwargs):
        """
        If you have trouble downloading HuggingFace models, -_^ this might help!!

        For Linux:
        export HF_ENDPOINT=https://hf-mirror.com

        For Windows:
        Good luck
        ^_-

        """
        super().__init__(key, model_name, **kwargs)
        if not cfg.light() and not DefaultRerank._model:
            import torch
            from FlagEmbedding import FlagReranker
            with DefaultRerank._model_lock:
                if not DefaultRerank._model:
                    try:
                        DefaultRerank._model = FlagReranker(
                            os.path.join(cfg.model_dir(), re.sub(
                                r"^[a-zA-Z]+/", "", model_name)),
                            use_fp16=torch.cuda.is_available())
                    except Exception as e:
                        model_dir = snapshot_download(repo_id=model_name,
                                                      local_dir=os.path.join(cfg.model_dir(),
                                                                             re.sub(r"^[a-zA-Z]+/", "", model_name)),
                                                      local_dir_use_symlinks=False)
                        DefaultRerank._model = FlagReranker(
                            model_dir, use_fp16=torch.cuda.is_available())
        self._model = DefaultRerank._model

    def similarity(self, query: str, texts: list):
        pairs = [(query, truncate(t, 2048)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 4096
        res = []
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(
                pairs[i:i + batch_size], max_length=2048)
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)
        return np.array(res), token_count


class JinaRerank(BaseRerank):
    def __init__(self, key, model_name="jina-reranker-v1-base-en",
                 base_url="https://api.jina.ai/v1/rerank", **kwargs):
        self.base_url = "https://api.jina.ai/v1/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        texts = [truncate(t, 8196) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts)
        }
        res = requests.post(
            self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, res["usage"]["total_tokens"]


class YoudaoRerank(DefaultRerank):
    _model = None
    _model_lock = threading.Lock()

    def __init__(self, key=None, model_name="maidalun1020/bce-reranker-base_v1", **kwargs):
        super().__init__(key, model_name, **kwargs)
        if not cfg.light() and not YoudaoRerank._model:
            from BCEmbedding import RerankerModel
            with YoudaoRerank._model_lock:
                if not YoudaoRerank._model:
                    try:
                        print("LOADING BCE...")
                        YoudaoRerank._model = RerankerModel(model_name_or_path=os.path.join(
                            self.models_dir,
                            re.sub(r"^[a-zA-Z]+/", "", model_name)))
                    except Exception as e:
                        YoudaoRerank._model = RerankerModel(
                            model_name_or_path=model_name.replace(
                                "maidalun1020", "InfiniFlow"))

        self._model = YoudaoRerank._model

    def similarity(self, query: str, texts: list):
        pairs = [(query, truncate(t, self._model.max_length)) for t in texts]
        token_count = 0
        for _, t in pairs:
            token_count += num_tokens_from_string(t)
        batch_size = 8
        res = []
        for i in range(0, len(pairs), batch_size):
            scores = self._model.compute_score(
                pairs[i:i + batch_size], max_length=self._model.max_length)
            scores = sigmoid(np.array(scores)).tolist()
            if isinstance(scores, float):
                res.append(scores)
            else:
                res.extend(scores)
        return np.array(res), token_count


class XInferenceRerank(BaseRerank):
    def __init__(self, key="xxxxxxx", model_name="", base_url="", **kwargs):
        if base_url.find("/v1") == -1:
            base_url = urljoin(base_url, "/v1/rerank")
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "accept": "application/json",
            "Authorization": f"Bearer {key}"
        }

    def similarity(self, query: str, texts: list):
        if len(texts) == 0:
            return np.array([]), 0
        data = {
            "model": self.model_name,
            "query": query,
            "return_documents": "true",
            "return_len": "true",
            "documents": texts
        }
        res = requests.post(
            self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, res["meta"]["tokens"]["input_tokens"] + res["meta"]["tokens"]["output_tokens"]


class LocalAIRerank(BaseRerank):
    def __init__(self, key, model_name, base_url):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("The LocalAIRerank has not been implement")


class NvidiaRerank(BaseRerank):
    def __init__(
            self, key, model_name, base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia/"
    ):
        if not base_url:
            base_url = "https://ai.api.nvidia.com/v1/retrieval/nvidia/"
        self.model_name = model_name

        if self.model_name == "nvidia/nv-rerankqa-mistral-4b-v3":
            self.base_url = os.path.join(
                base_url, "nv-rerankqa-mistral-4b-v3", "reranking"
            )

        if self.model_name == "nvidia/rerank-qa-mistral-4b":
            self.base_url = os.path.join(base_url, "reranking")
            self.model_name = "nv-rerank-qa-mistral-4b:1"

        self.headers = {
            "accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    def similarity(self, query: str, texts: list):
        token_count = num_tokens_from_string(query) + sum(
            [num_tokens_from_string(t) for t in texts]
        )
        data = {
            "model": self.model_name,
            "query": {"text": query},
            "passages": [{"text": text} for text in texts],
            "truncate": "END",
            "top_n": len(texts),
        }
        res = requests.post(
            self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        for d in res["rankings"]:
            rank[d["index"]] = d["logit"]
        return rank, token_count


class LmStudioRerank(BaseRerank):
    def __init__(self, key, model_name, base_url):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("The LmStudioRerank has not been implement")


class OpenAI_APIRerank(BaseRerank):
    def __init__(self, key, model_name, base_url):
        if base_url.find("/rerank") == -1:
            self.base_url = urljoin(base_url, "/rerank")
        else:
            self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        # noway to config Ragflow , use fix setting
        texts = [truncate(t, 500) for t in texts]
        data = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
        token_count = 0
        for t in texts:
            token_count += num_tokens_from_string(t)
        res = requests.post(
            self.base_url, headers=self.headers, json=data).json()
        rank = np.zeros(len(texts), dtype=float)
        if 'results' not in res:
            raise ValueError("response not contains results\n" + str(res))
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]

        # Normalize the rank values to the range 0 to 1
        min_rank = np.min(rank)
        max_rank = np.max(rank)

        # Avoid division by zero if all ranks are identical
        if max_rank - min_rank != 0:
            rank = (rank - min_rank) / (max_rank - min_rank)
        else:
            rank = np.zeros_like(rank)

        return rank, token_count


class CoHereRerank(BaseRerank):
    def __init__(self, key, model_name, base_url=None):

        self.client = CoClient(api_key=key)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        token_count = num_tokens_from_string(query) + sum(
            [num_tokens_from_string(t) for t in texts]
        )
        res = self.client.rerank(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False,
        )
        rank = np.zeros(len(texts), dtype=float)
        for d in res.results:
            rank[d.index] = d.relevance_score
        return rank, token_count


class TogetherAIRerank(BaseRerank):
    def __init__(self, key, model_name, base_url):
        pass

    def similarity(self, query: str, texts: list):
        raise NotImplementedError("The api has not been implement")


class SILICONFLOWRerank(BaseRerank):
    def __init__(
            self, key, model_name, base_url="https://api.siliconflow.cn/v1/rerank"
    ):
        if not base_url:
            base_url = "https://api.siliconflow.cn/v1/rerank"
        self.model_name = model_name
        self.base_url = base_url
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {key}",
        }

    def similarity(self, query: str, texts: list):
        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
            "return_documents": False,
            "max_chunks_per_doc": 1024,
            "overlap_tokens": 80,
        }
        response = requests.post(
            self.base_url, json=payload, headers=self.headers
        ).json()
        rank = np.zeros(len(texts), dtype=float)
        if "results" not in response:
            return rank, 0

        for d in response["results"]:
            rank[d["index"]] = d["relevance_score"]
        return (
            rank,
            response["meta"]["tokens"]["input_tokens"] +
            response["meta"]["tokens"]["output_tokens"],
        )


class BaiduYiyanRerank(BaseRerank):
    def __init__(self, key, model_name, base_url=None):

        key = json.loads(key)
        ak = key.get("yiyan_ak", "")
        sk = key.get("yiyan_sk", "")
        self.client = Reranker(ak=ak, sk=sk)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        res = self.client.do(
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
        ).body
        rank = np.zeros(len(texts), dtype=float)
        for d in res["results"]:
            rank[d["index"]] = d["relevance_score"]
        return rank, res["usage"]["total_tokens"]


class VoyageRerank(BaseRerank):
    def __init__(self, key, model_name, base_url=None):

        self.client = voyageai.Client(api_key=key)
        self.model_name = model_name

    def similarity(self, query: str, texts: list):
        res = self.client.rerank(
            query=query, documents=texts, model=self.model_name, top_k=len(
                texts)
        )
        rank = np.zeros(len(texts), dtype=float)
        for r in res.results:
            rank[r.index] = r.relevance_score
        return rank, res.total_tokens


class QWenRerank(BaseRerank):
    def __init__(self, key, model_name='gte-rerank', base_url=None, **kwargs):
        self.api_key = key
        self.model_name = dashscope.TextReRank.Models.gte_rerank if model_name is None else model_name

    def similarity(self, query: str, texts: list):
        from http import HTTPStatus

        import dashscope
        resp = dashscope.TextReRank.call(
            api_key=self.api_key,
            model=self.model_name,
            query=query,
            documents=texts,
            top_n=len(texts),
            return_documents=False
        )
        rank = np.zeros(len(texts), dtype=float)
        if resp.status_code == HTTPStatus.OK:
            for r in resp.output.results:
                rank[r.index] = r.relevance_score
            return rank, resp.usage.total_tokens
        return rank, 0


RerankModel = {
    "BAAI": DefaultRerank,
    "Jina": JinaRerank,
    "Youdao": YoudaoRerank,
    "Xinference": XInferenceRerank,
    "NVIDIA": NvidiaRerank,
    "LM-Studio": LmStudioRerank,
    "OpenAI-API-Compatible": OpenAI_APIRerank,
    "cohere": CoHereRerank,
    "TogetherAI": TogetherAIRerank,
    "SILICONFLOW": SILICONFLOWRerank,
    "BaiduYiyan": BaiduYiyanRerank,
    "Voyage AI": VoyageRerank,
    "Tongyi-Qianwen": QWenRerank,
}
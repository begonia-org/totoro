#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   embedding.py
@Time    :   2024/11/01 09:53:02
@Desc    :   
'''


import os
import re
from abc import ABC
from typing import Optional

import dashscope
import numpy as np
import requests
import torch
from BCEmbedding import EmbeddingModel as qanthing
from fastembed import TextEmbedding
from FlagEmbedding import FlagModel
from huggingface_hub import snapshot_download
from ollama import Client
from openai import OpenAI
from zhipuai import ZhipuAI

from totoro.utils.encoder import ModelEncoder
from totoro.config import TotoroConfigure as cfg
from totoro.utils.logger import nlp_logger


class BaseEmbedding(ABC):
    def __init__(self, key, model_name, models_dir=cfg.model_dir()):
        """The base class for embedding models.
        Args:
            key (str): The API key for the embedding model.
            model_name (str): The name of the model.
            models_dir (str): The directory to save the model.
        """
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.model_encoder = ModelEncoder()

    def encode(self, texts: list, batch_size=32):
        raise NotImplementedError("Please implement encode method!")

    def encode_queries(self, text: str):
        raise NotImplementedError("Please implement encode method!")


class DefaultEmbedding(BaseEmbedding):
    _model = None

    def __init__(self, key, model_name, **kwargs):
        """
        """
        super().__init__(key, model_name, **kwargs)
        self.model_encoder = ModelEncoder()
        nlp_logger.debug(f"Loading BAAI model {model_name}...")
        if not DefaultEmbedding._model:
            try:
                path = os.path.join(
                    self.models_dir,
                    re.sub(
                        r"^[a-zA-Z]+/",
                        "",
                        model_name))
                nlp_logger.debug(f"model path:{path}")
                self._model = FlagModel(
                    os.path.join(
                        self.models_dir,
                        re.sub(
                            r"^[a-zA-Z]+/",
                            "",
                            model_name)),
                    query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                    use_fp16=torch.cuda.is_available())
            except Exception:
                model_dir = snapshot_download(
                    repo_id="BAAI/bge-large-zh-v1.5",
                    local_dir=os.path.join(
                        self.models_dir,
                        re.sub(
                            r"^[a-zA-Z]+/",
                            "",
                            model_name)),
                    local_dir_use_symlinks=False)
                self._model = FlagModel(model_dir,
                                        query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                                        use_fp16=torch.cuda.is_available())

    def encode(self, texts: list, batch_size=32):
        texts = [self.model_encoder.truncate(t, 2048) for t in texts]
        token_count = 0
        for t in texts:
            token_count += self.model_encoder.num_tokens_from_string(t)
        res = []
        for i in range(0, len(texts), batch_size):
            res.extend(self._model.encode(texts[i:i + batch_size]).tolist())

        # nlp_logger.debug(traceback.format_stack())
        # nlp_logger.debug(f'"batch_size":{batch_size},texts":"{texts}","token_count":{token_count},"result":"{np.array(res).tolist()}"')

        return np.array(res), token_count

    def encode_queries(self, text: str):
        token_count = self.model_encoder.num_tokens_from_string(text)
        return self._model.encode_queries([text]).tolist()[0], token_count


class OpenAIEmbed(BaseEmbedding):
    def __init__(self, key, model_name="text-embedding-ada-002",
                 base_url="https://api.openai.com/v1"):
        super().__init__(key, model_name)

        if not base_url:
            base_url = "https://api.openai.com/v1"
        self.client = OpenAI(api_key=key, base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        texts = [self.model_encoder.truncate(t, 8196) for t in texts]
        res = self.client.embeddings.create(input=texts,
                                            model=self.model_name)
        return np.array([d.embedding for d in res.data]
                        ), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=[self.model_encoder.truncate(text, 8196)],
                                            model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class BaiChuanEmbed(OpenAIEmbed):
    def __init__(self, key,
                 model_name='Baichuan-Text-Embedding',
                 base_url='https://api.baichuan-ai.com/v1'):
        if not base_url:
            base_url = "https://api.baichuan-ai.com/v1"
        super().__init__(key, model_name, base_url)


class QWenEmbed(BaseEmbedding):
    def __init__(self, key, model_name="text_embedding_v2", **kwargs):
        dashscope.api_key = key
        self.model_name = model_name

    def encode(self, texts: list, batch_size=10):
        try:
            res = []
            token_count = 0
            texts = [self.model_encoder.truncate(t, 2048) for t in texts]
            for i in range(0, len(texts), batch_size):
                resp = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=texts[i:i + batch_size],
                    text_type="document"
                )
                embds = [[] for _ in range(len(resp["output"]["embeddings"]))]
                for e in resp["output"]["embeddings"]:
                    embds[e["text_index"]] = e["embedding"]
                res.extend(embds)
                token_count += resp["usage"]["total_tokens"]
            return np.array(res), token_count
        except Exception as e:
            raise Exception(
                "Account abnormal. Please ensure it's on good standing.")

    def encode_queries(self, text):
        try:
            resp = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=text[:2048],
                text_type="query"
            )
            return np.array(resp["output"]["embeddings"][0]
                            ["embedding"]), resp["usage"]["total_tokens"]
        except Exception as e:
            raise Exception(
                "Account abnormal. Please ensure it's on good standing.")


class ZhipuEmbed(BaseEmbedding):
    def __init__(self, key, model_name="embedding-2", **kwargs):
        self.client = ZhipuAI(api_key=key)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        for txt in texts:
            res = self.client.embeddings.create(input=txt,
                                                model=self.model_name)
            arr.append(res.data[0].embedding)
            tks_num += res.usage.total_tokens
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=text,
                                            model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class OllamaEmbed(BaseEmbedding):
    def __init__(self, key, model_name, **kwargs):
        self.client = Client(host=kwargs["base_url"])
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        arr = []
        tks_num = 0
        for txt in texts:
            res = self.client.embeddings(prompt=txt,
                                         model=self.model_name)
            arr.append(res["embedding"])
            tks_num += 128
        return np.array(arr), tks_num

    def encode_queries(self, text):
        res = self.client.embeddings(prompt=text,
                                     model=self.model_name)
        return np.array(res["embedding"]), 128


class FastEmbed(BaseEmbedding):
    _model = None

    def __init__(
            self,
            key: Optional[str] = None,
            model_name: str = "BAAI/bge-small-en-v1.5",
            cache_dir: Optional[str] = None,
            threads: Optional[int] = None,
            **kwargs,
    ):
        if not FastEmbed._model:
            self._model = TextEmbedding(
                model_name, cache_dir, threads, **kwargs)

    def encode(self, texts: list, batch_size=32):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encodings = self._model.model.tokenizer.encode_batch(texts)
        total_tokens = sum(len(e) for e in encodings)

        embeddings = [e.tolist() for e in self._model.embed(texts, batch_size)]

        return np.array(embeddings), total_tokens

    def encode_queries(self, text: str):
        # Using the internal tokenizer to encode the texts and get the total
        # number of tokens
        encoding = self._model.model.tokenizer.encode(text)
        embedding = next(self._model.query_embed(text)).tolist()

        return np.array(embedding), len(encoding.ids)


class XinferenceEmbed(BaseEmbedding):
    def __init__(self, key, model_name="", base_url=""):
        self.client = OpenAI(api_key="xxx", base_url=base_url)
        self.model_name = model_name

    def encode(self, texts: list, batch_size=32):
        res = self.client.embeddings.create(input=texts,
                                            model=self.model_name)
        return np.array([d.embedding for d in res.data]
                        ), res.usage.total_tokens

    def encode_queries(self, text):
        res = self.client.embeddings.create(input=[text],
                                            model=self.model_name)
        return np.array(res.data[0].embedding), res.usage.total_tokens


class YoudaoEmbed(BaseEmbedding):
    _client = None

    def __init__(self, key=None, model_name="maidalun1020/bce-embedding-base_v1", **kwargs):
        if not YoudaoEmbed._client:
            try:
                nlp_logger.debug("LOADING BCE...")
                YoudaoEmbed._client = qanthing(model_name_or_path=os.path.join(
                    self.models_dir,
                    "bce-embedding-base_v1"))
            except Exception as e:
                YoudaoEmbed._client = qanthing(
                    model_name_or_path=model_name.replace(
                        "maidalun1020", "InfiniFlow"))

    def encode(self, texts: list, batch_size=10):
        res = []
        token_count = 0
        for t in texts:
            token_count += self.model_encoder.num_tokens_from_string(t)
        for i in range(0, len(texts), batch_size):
            embds = YoudaoEmbed._client.encode(texts[i:i + batch_size])
            res.extend(embds)
        return np.array(res), token_count

    def encode_queries(self, text):
        embds = YoudaoEmbed._client.encode([text])
        return np.array(embds[0]), self.model_encoder.num_tokens_from_string(text)


class JinaEmbed(BaseEmbedding):
    def __init__(self, key, model_name="jina-embeddings-v2-base-zh",
                 base_url="https://api.jina.ai/v1/embeddings"):

        self.base_url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}"
        }
        self.model_name = model_name

    def encode(self, texts: list, batch_size=None):
        texts = [self.model_encoder.truncate(t, 8196) for t in texts]
        data = {
            "model": self.model_name,
            "input": texts,
            'encoding_type': 'float'
        }
        res = requests.post(
            self.base_url, headers=self.headers, json=data).json()
        return np.array([d["embedding"] for d in res["data"]]), res["usage"]["total_tokens"]

    def encode_queries(self, text):
        embds, cnt = self.encode([text])
        return np.array(embds[0]), cnt


EmbeddingModel = {
    "Ollama": OllamaEmbed,
    "OpenAI": OpenAIEmbed,
    "Xinference": XinferenceEmbed,
    "Tongyi-Qianwen": QWenEmbed,
    "ZHIPU-AI": ZhipuEmbed,
    "FastEmbed": FastEmbed,
    "Youdao": YoudaoEmbed,
    "BaiChuan": BaiChuanEmbed,
    "Jina": JinaEmbed,
    "BAAI": DefaultEmbedding
}

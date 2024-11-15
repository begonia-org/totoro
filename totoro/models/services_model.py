
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   services.py
@Time    :
@Desc    :
'''


from pydantic import BaseModel

from typing import List, Optional, Type

from pydantic import Field as _Field

from .doc_model import EmbededItem

from typing import Dict, List, Optional, Type

from .doc_model import ParserConfig

from .doc_model import DocSearchVector

from pydantic_protobuf.ext import model2protobuf, pool, PydanticModel, protobuf2model

from google.protobuf import message_factory

from google.protobuf import message as _message

from typing import Dict, Optional, Type, List, Any

from .constant_model import ChunkType


class EmbeddingRequest(BaseModel):

    file_key_or_url: Optional[str] = _Field()
    lang: Optional[str] = _Field()
    chunk_type: Optional[ChunkType] = _Field()
    parser_config: Optional[ParserConfig] = _Field()
    embedding: Optional[str] = _Field()
    model_api_key: Optional[str] = _Field()
    doc_title: Optional[str] = _Field()
    task_id: Optional[str] = _Field()
    important_keywords: Optional[List[str]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.EmbeddingRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class EmbeddingResponse(BaseModel):

    items: Optional[List[EmbededItem]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.EmbeddingResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class EmbeddingProgressRequest(BaseModel):

    task_id: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.EmbeddingProgressRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CandidateTokens(BaseModel):

    token: Optional[str] = _Field()
    doc_id: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.CandidateTokens")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ReankingRequest(BaseModel):

    query: Optional[str] = _Field()
    candidates: Optional[Dict[str, str]] = _Field()
    rerank: Optional[str] = _Field()
    model_api_key: Optional[str] = _Field()
    keyword_simlarity_weight: Optional[float] = _Field()
    semantic_simlarity_weight: Optional[float] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.ReankingRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ReankingResponse(BaseModel):

    weighted_similarity_ranks: Optional[Dict[str, Any]] = _Field()
    keyword_similarity_ranks: Optional[Dict[str, Any]] = _Field()
    semantic_similarity_ranks: Optional[Dict[str, Any]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.ReankingResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class QueryBuildRequest(BaseModel):

    query: Optional[str] = _Field()
    top: Optional[int] = _Field()
    simlarity_threshold: Optional[float] = _Field()
    embedding: Optional[str] = _Field()
    model_api_key: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.QueryBuildRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class QueryBuildResponse(BaseModel):

    vector: Optional[DocSearchVector] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.QueryBuildResponse")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   doc.py
@Time    :
@Desc    :
'''


from pydantic import Field as _Field

from google.protobuf import message as _message

from pydantic_protobuf.ext import pool, protobuf2model, PydanticModel, model2protobuf

from pydantic import BaseModel

from typing import Optional, Dict, Type, List, Any

from .constant_model import ChunkType

from typing import Optional, Type

from typing import Optional, Type, List

from google.protobuf import message_factory


class Position(BaseModel):

    number: Optional[int] = _Field()
    left: Optional[int] = _Field()
    right: Optional[int] = _Field()
    top: Optional[int] = _Field()
    bottom: Optional[int] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.Position")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class ParserConfig(BaseModel):

    filename_embd_weight: Optional[float] = _Field(description="filename_embd_weight", default=0.2)
    task_page_size: Optional[int] = _Field(description="task_page_size", default=12)
    do_layout_recognize: Optional[bool] = _Field(description="do_layout_recognize", default=True)
    chunk_token_num: Optional[int] = _Field(description="chunk_token_num", default=128)
    delimiter: Optional[str] = _Field(description="delimiter", default="\n!?。；！？")

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.ParserConfig")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocTask(BaseModel):

    file_id: Optional[str] = _Field()
    lang: Optional[str] = _Field()
    llm_id: Optional[str] = _Field()
    kb_id: Optional[str] = _Field()
    chunk_type: Optional[ChunkType] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocTask")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class Doc(BaseModel):

    doc_id: Optional[str] = _Field()
    doc_name_keyword: Optional[str] = _Field()
    title_tokens: Optional[str] = _Field()
    title_small_tokens: Optional[str] = _Field()
    page_num: Optional[List[int]] = _Field()
    positions: Optional[List[Position]] = _Field()
    top: Optional[List[int]] = _Field()
    content_with_weight: Optional[str] = _Field()
    content_tokens: Optional[str] = _Field()
    content_small_tokens: Optional[str] = _Field()
    image_uri: Optional[str] = _Field()
    q_vec: Optional[List[float]] = _Field()
    q_vec_size: Optional[int] = _Field()
    image: Optional[str] = _Field()
    important_keywords: Optional[List[str]] = _Field()
    important_keywords_tokens: Optional[str] = _Field()
    file_key: Optional[str] = _Field()
    file_md5: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.Doc")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class Smilarity(BaseModel):

    simlarity: Optional[List[float]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.Smilarity")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class EmbededItem(BaseModel):

    doc: Optional[Doc] = _Field()
    tokens: Optional[int] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.EmbededItem")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocItem(BaseModel):

    doc_id: Optional[str] = _Field()
    kb_ids: Optional[List[str]] = _Field()
    doc_name_keyword: Optional[str] = _Field()
    title_tokens: Optional[str] = _Field()
    positions: Optional[List[Position]] = _Field()
    content_with_weight: Optional[str] = _Field(description="content_with_weight", default="")
    content_tokens: Optional[str] = _Field()
    image: Optional[str] = _Field()
    q_vec_size: Optional[int] = _Field()
    important_keyword: Optional[List[str]] = _Field(description="important_keyword", default=[])
    q_vec: Optional[List[float]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocItem")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocAggs(BaseModel):

    doc_name: Optional[str] = _Field()
    doc_id: Optional[str] = _Field()
    count: Optional[int] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocAggs")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocChunk(BaseModel):

    doc_id: Optional[str] = _Field()
    kb_ids: Optional[List[str]] = _Field()
    important_keyword: Optional[List[str]] = _Field()
    doc_name_keyword: Optional[str] = _Field()
    image: Optional[str] = _Field()
    similarity: Optional[float] = _Field()
    vector_similarity: Optional[float] = _Field()
    term_similarity: Optional[float] = _Field()
    content_with_weight: Optional[str] = _Field()
    content_tokens: Optional[str] = _Field()
    positions: Optional[List[Position]] = _Field()
    chunk_id: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocChunk")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocDegreeProgress(BaseModel):

    message: Optional[str] = _Field()
    progress: Optional[float] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocDegreeProgress")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class SearchRequest(BaseModel):

    kb_ids: Optional[List[str]] = _Field()
    doc_ids: Optional[List[str]] = _Field()
    size: Optional[int] = _Field()
    question: Optional[str] = _Field()
    vector: Optional[bool] = _Field()
    topk: Optional[int] = _Field()
    similarity: Optional[float] = _Field()
    available: Optional[bool] = _Field()
    page: Optional[int] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.SearchRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocSearchResult(BaseModel):

    total: Optional[int] = _Field()
    ids: Optional[List[str]] = _Field()
    query_vector: Optional[List[float]] = _Field()
    docs: Optional[Dict[str, Any]] = _Field()
    highlight: Optional[Dict[str, str]] = _Field()
    aggregation: Optional[List[str]] = _Field()
    keywords: Optional[List[str]] = _Field()
    group_docs: Optional[List[str]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocSearchResult")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class DocSearchVector(BaseModel):

    q_vec_size: Optional[int] = _Field()
    top_k: Optional[int] = _Field()
    similarity: Optional[float] = _Field()
    num_candidates: Optional[int] = _Field()
    query_vector: Optional[List[float]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.DocSearchVector")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class TokenizerItem(BaseModel):

    token: Optional[str] = _Field()
    synonyms: Optional[List[str]] = _Field()
    weight: Optional[float] = _Field()
    fine_grained_tokens: Optional[List[str]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.TokenizerItem")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class TokenWeight(BaseModel):

    token: Optional[str] = _Field()
    weight: Optional[float] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.TokenWeight")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class TermWeightTokens(BaseModel):

    tokens: Optional[List[str]] = _Field()
    weight: Optional[float] = _Field()
    synonyms_tokens: Optional[List[str]] = _Field()
    sorted_weight_tokens: Optional[List[TokenizerItem]] = _Field()
    isalnum: Optional[bool] = _Field()
    synonyms: Optional[List[str]] = _Field()
    token_weights: Optional[List[TokenWeight]] = _Field()
    is_chinese: Optional[bool] = _Field()
    term_weight_spilt_str: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.TermWeightTokens")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)

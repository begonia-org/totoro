from totoro.pb import constant_pb2 as _constant_pb2
from totoro.pb import doc_pb2 as _doc_pb2
from pydantic_protobuf import pydantic_pb2 as _pydantic_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EmbeddingRequest(_message.Message):
    __slots__ = ("file_key_or_url", "lang", "chunk_type", "parser_config", "embedding", "model_api_key", "doc_title", "task_id", "important_keywords")
    FILE_KEY_OR_URL_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    CHUNK_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    MODEL_API_KEY_FIELD_NUMBER: _ClassVar[int]
    DOC_TITLE_FIELD_NUMBER: _ClassVar[int]
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    file_key_or_url: str
    lang: str
    chunk_type: _constant_pb2.ChunkType
    parser_config: _doc_pb2.ParserConfig
    embedding: str
    model_api_key: str
    doc_title: str
    task_id: str
    important_keywords: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, file_key_or_url: _Optional[str] = ..., lang: _Optional[str] = ..., chunk_type: _Optional[_Union[_constant_pb2.ChunkType, str]] = ..., parser_config: _Optional[_Union[_doc_pb2.ParserConfig, _Mapping]] = ..., embedding: _Optional[str] = ..., model_api_key: _Optional[str] = ..., doc_title: _Optional[str] = ..., task_id: _Optional[str] = ..., important_keywords: _Optional[_Iterable[str]] = ...) -> None: ...

class EmbeddingResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[_doc_pb2.EmbededItem]
    def __init__(self, items: _Optional[_Iterable[_Union[_doc_pb2.EmbededItem, _Mapping]]] = ...) -> None: ...

class EmbeddingProgressRequest(_message.Message):
    __slots__ = ("task_id",)
    TASK_ID_FIELD_NUMBER: _ClassVar[int]
    task_id: str
    def __init__(self, task_id: _Optional[str] = ...) -> None: ...

class CandidateTokens(_message.Message):
    __slots__ = ("token", "doc_id")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    token: str
    doc_id: str
    def __init__(self, token: _Optional[str] = ..., doc_id: _Optional[str] = ...) -> None: ...

class ReankingRequest(_message.Message):
    __slots__ = ("query", "candidates", "rerank", "model_api_key", "keyword_simlarity_weight", "semantic_simlarity_weight")
    class CandidatesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    QUERY_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    RERANK_FIELD_NUMBER: _ClassVar[int]
    MODEL_API_KEY_FIELD_NUMBER: _ClassVar[int]
    KEYWORD_SIMLARITY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_SIMLARITY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    query: str
    candidates: _containers.ScalarMap[str, str]
    rerank: str
    model_api_key: str
    keyword_simlarity_weight: float
    semantic_simlarity_weight: float
    def __init__(self, query: _Optional[str] = ..., candidates: _Optional[_Mapping[str, str]] = ..., rerank: _Optional[str] = ..., model_api_key: _Optional[str] = ..., keyword_simlarity_weight: _Optional[float] = ..., semantic_simlarity_weight: _Optional[float] = ...) -> None: ...

class ReankingResponse(_message.Message):
    __slots__ = ("weighted_similarity_ranks", "keyword_similarity_ranks", "semantic_similarity_ranks")
    class WeightedSimilarityRanksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _doc_pb2.Smilarity
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_doc_pb2.Smilarity, _Mapping]] = ...) -> None: ...
    class KeywordSimilarityRanksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _doc_pb2.Smilarity
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_doc_pb2.Smilarity, _Mapping]] = ...) -> None: ...
    class SemanticSimilarityRanksEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _doc_pb2.Smilarity
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_doc_pb2.Smilarity, _Mapping]] = ...) -> None: ...
    WEIGHTED_SIMILARITY_RANKS_FIELD_NUMBER: _ClassVar[int]
    KEYWORD_SIMILARITY_RANKS_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_SIMILARITY_RANKS_FIELD_NUMBER: _ClassVar[int]
    weighted_similarity_ranks: _containers.MessageMap[str, _doc_pb2.Smilarity]
    keyword_similarity_ranks: _containers.MessageMap[str, _doc_pb2.Smilarity]
    semantic_similarity_ranks: _containers.MessageMap[str, _doc_pb2.Smilarity]
    def __init__(self, weighted_similarity_ranks: _Optional[_Mapping[str, _doc_pb2.Smilarity]] = ..., keyword_similarity_ranks: _Optional[_Mapping[str, _doc_pb2.Smilarity]] = ..., semantic_similarity_ranks: _Optional[_Mapping[str, _doc_pb2.Smilarity]] = ...) -> None: ...

class QueryBuildRequest(_message.Message):
    __slots__ = ("query", "top", "simlarity_threshold", "embedding", "model_api_key")
    QUERY_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    SIMLARITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    MODEL_API_KEY_FIELD_NUMBER: _ClassVar[int]
    query: str
    top: int
    simlarity_threshold: float
    embedding: str
    model_api_key: str
    def __init__(self, query: _Optional[str] = ..., top: _Optional[int] = ..., simlarity_threshold: _Optional[float] = ..., embedding: _Optional[str] = ..., model_api_key: _Optional[str] = ...) -> None: ...

class QueryBuildResponse(_message.Message):
    __slots__ = ("vector",)
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    vector: _doc_pb2.DocSearchVector
    def __init__(self, vector: _Optional[_Union[_doc_pb2.DocSearchVector, _Mapping]] = ...) -> None: ...

class PreQuestionRequest(_message.Message):
    __slots__ = ("question",)
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    question: str
    def __init__(self, question: _Optional[str] = ...) -> None: ...

class PreQuestionResponse(_message.Message):
    __slots__ = ("term_weight_tokens",)
    TERM_WEIGHT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    term_weight_tokens: _doc_pb2.TermWeightTokens
    def __init__(self, term_weight_tokens: _Optional[_Union[_doc_pb2.TermWeightTokens, _Mapping]] = ...) -> None: ...

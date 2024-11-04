import constant_pb2 as _constant_pb2
import doc_pb2 as _doc_pb2
import pydantic_pb2 as _pydantic_pb2
from google.api import annotations_pb2 as _annotations_pb2
from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EmbeddingRequest(_message.Message):
    __slots__ = ("file_key", "lang", "chunk_type", "parser_config", "embedding")
    FILE_KEY_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    CHUNK_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARSER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_FIELD_NUMBER: _ClassVar[int]
    file_key: str
    lang: str
    chunk_type: _constant_pb2.ChunkType
    parser_config: _doc_pb2.ParserConfig
    embedding: str
    def __init__(self, file_key: _Optional[str] = ..., lang: _Optional[str] = ..., chunk_type: _Optional[_Union[_constant_pb2.ChunkType, str]] = ..., parser_config: _Optional[_Union[_doc_pb2.ParserConfig, _Mapping]] = ..., embedding: _Optional[str] = ...) -> None: ...

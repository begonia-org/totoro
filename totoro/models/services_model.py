
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   services.py
@Time    :
@Desc    :
'''


from .doc_model import ParserConfig

from pydantic_protobuf.ext import protobuf2model, PydanticModel, pool, model2protobuf

from google.protobuf import message as _message

from pydantic import Field as _Field

from pydantic import BaseModel

from .constant_model import ChunkType

from typing import Optional, Type

from google.protobuf import message_factory


class EmbeddingRequest(BaseModel):

    file_key: Optional[str] = _Field()
    lang: Optional[str] = _Field()
    chunk_type: Optional[ChunkType] = _Field()
    parser_config: Optional[ParserConfig] = _Field()
    embedding: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("totoro.EmbeddingRequest")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)

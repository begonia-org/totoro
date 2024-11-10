
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   http.py
@Time    :
@Desc    :
'''


from pydantic import BaseModel

from pydantic import Field as _Field

from typing import Optional, List, Type

from google.protobuf import message as _message

from pydantic_protobuf.ext import pool, model2protobuf, PydanticModel, protobuf2model

from google.protobuf import message_factory


class Http(BaseModel):

    rules: Optional[List[HttpRule]] = _Field()
    fully_decode_reserved_expansion: Optional[bool] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("google.api.Http")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class HttpRule(BaseModel):

    selector: Optional[str] = _Field()
    get: Optional[str] = _Field()
    put: Optional[str] = _Field()
    post: Optional[str] = _Field()
    delete: Optional[str] = _Field()
    patch: Optional[str] = _Field()
    custom: Optional[CustomHttpPattern] = _Field()
    body: Optional[str] = _Field()
    response_body: Optional[str] = _Field()
    additional_bindings: Optional[List[HttpRule]] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("google.api.HttpRule")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)


class CustomHttpPattern(BaseModel):

    kind: Optional[str] = _Field()
    path: Optional[str] = _Field()

    def to_protobuf(self) -> _message.Message:
        _proto = pool.FindMessageTypeByName("google.api.CustomHttpPattern")
        _cls: Type[_message.Message] = message_factory.GetMessageClass(_proto)
        return model2protobuf(self, _cls())

    @classmethod
    def from_protobuf(cls: Type[PydanticModel], src: _message.Message) -> PydanticModel:
        return protobuf2model(cls, src)

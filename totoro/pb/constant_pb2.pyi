from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class ChunkType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CHUNK_TYPE_UNKNOWN: _ClassVar[ChunkType]
    CHUNK_TYPE_NAIVE: _ClassVar[ChunkType]
    CHUNK_TYPE_PAPER: _ClassVar[ChunkType]

class FileType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FILE_TYPE_UNKNOWN: _ClassVar[FileType]
    FILE_TYPE_PDF: _ClassVar[FileType]
    FILE_TYPE_DOC: _ClassVar[FileType]
    FILE_TYPE_VISUAL: _ClassVar[FileType]
    FILE_TYPE_AURAL: _ClassVar[FileType]
    FILE_TYPE_VIRTUAL: _ClassVar[FileType]
    FILE_TYPE_FOLDER: _ClassVar[FileType]
    FILE_TYPE_OTHER: _ClassVar[FileType]

class LLMType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LLM_TYPE_UNKNOWN: _ClassVar[LLMType]
    LLM_TYPE_CHAT: _ClassVar[LLMType]
    LLM_TYPE_EMBEDDING: _ClassVar[LLMType]
    LLM_TYPE_SPEECH2TEXT: _ClassVar[LLMType]
    LLM_TYPE_IMAGE2TEXT: _ClassVar[LLMType]
    LLM_TYPE_RERANK: _ClassVar[LLMType]

class LLMStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LLM_STATUS_UNKNOWN: _ClassVar[LLMStatus]
    LLM_STATUS_ACTIVE: _ClassVar[LLMStatus]
    LLM_STATUS_INACTIVE: _ClassVar[LLMStatus]
    LLM_STATUS_DELETED: _ClassVar[LLMStatus]

class PromptType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SIMPLE: _ClassVar[PromptType]
    ADVANCED: _ClassVar[PromptType]

class AssistantStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    AS_WASTED: _ClassVar[AssistantStatus]
    AS_VALIDATE: _ClassVar[AssistantStatus]

class ConversationRole(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    CONVE_ASSISTANT: _ClassVar[ConversationRole]
    CONVE_USER: _ClassVar[ConversationRole]
    CONVE_SYSTEM: _ClassVar[ConversationRole]
CHUNK_TYPE_UNKNOWN: ChunkType
CHUNK_TYPE_NAIVE: ChunkType
CHUNK_TYPE_PAPER: ChunkType
FILE_TYPE_UNKNOWN: FileType
FILE_TYPE_PDF: FileType
FILE_TYPE_DOC: FileType
FILE_TYPE_VISUAL: FileType
FILE_TYPE_AURAL: FileType
FILE_TYPE_VIRTUAL: FileType
FILE_TYPE_FOLDER: FileType
FILE_TYPE_OTHER: FileType
LLM_TYPE_UNKNOWN: LLMType
LLM_TYPE_CHAT: LLMType
LLM_TYPE_EMBEDDING: LLMType
LLM_TYPE_SPEECH2TEXT: LLMType
LLM_TYPE_IMAGE2TEXT: LLMType
LLM_TYPE_RERANK: LLMType
LLM_STATUS_UNKNOWN: LLMStatus
LLM_STATUS_ACTIVE: LLMStatus
LLM_STATUS_INACTIVE: LLMStatus
LLM_STATUS_DELETED: LLMStatus
SIMPLE: PromptType
ADVANCED: PromptType
AS_WASTED: AssistantStatus
AS_VALIDATE: AssistantStatus
CONVE_ASSISTANT: ConversationRole
CONVE_USER: ConversationRole
CONVE_SYSTEM: ConversationRole

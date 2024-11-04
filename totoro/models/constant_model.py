
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   constant.py
@Time    :
@Desc    :
'''


from enum import Enum as _Enum


class ChunkType(_Enum):
    CHUNK_TYPE_UNKNOWN = 0
    CHUNK_TYPE_NAIVE = 1
    CHUNK_TYPE_PAPER = 2


class FileType(_Enum):
    FILE_TYPE_UNKNOWN = 0
    FILE_TYPE_PDF = 1
    FILE_TYPE_DOC = 2
    FILE_TYPE_VISUAL = 3
    FILE_TYPE_AURAL = 4
    FILE_TYPE_VIRTUAL = 5
    FILE_TYPE_FOLDER = 6
    FILE_TYPE_OTHER = 7


class LLMType(_Enum):
    LLM_TYPE_UNKNOWN = 0
    LLM_TYPE_CHAT = 1
    LLM_TYPE_EMBEDDING = 2
    LLM_TYPE_SPEECH2TEXT = 3
    LLM_TYPE_IMAGE2TEXT = 4
    LLM_TYPE_RERANK = 5


class LLMStatus(_Enum):
    LLM_STATUS_UNKNOWN = 0
    LLM_STATUS_ACTIVE = 1
    LLM_STATUS_INACTIVE = 2
    LLM_STATUS_DELETED = 3


class PromptType(_Enum):
    SIMPLE = 0
    ADVANCED = 1


class AssistantStatus(_Enum):
    AS_WASTED = 0
    AS_VALIDATE = 1


class ConversationRole(_Enum):
    CONVE_ASSISTANT = 0
    CONVE_USER = 1
    CONVE_SYSTEM = 2

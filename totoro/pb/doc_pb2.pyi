from google.protobuf import any_pb2 as _any_pb2
from pydantic_protobuf import pydantic_pb2 as _pydantic_pb2
from totoro.pb import constant_pb2 as _constant_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Position(_message.Message):
    __slots__ = ("number", "left", "right", "top", "bottom")
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    number: int
    left: int
    right: int
    top: int
    bottom: int
    def __init__(self, number: _Optional[int] = ..., left: _Optional[int] = ..., right: _Optional[int] = ..., top: _Optional[int] = ..., bottom: _Optional[int] = ...) -> None: ...

class ParserConfig(_message.Message):
    __slots__ = ("filename_embd_weight", "task_page_size", "do_layout_recognize", "chunk_token_num", "delimiter")
    FILENAME_EMBD_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    TASK_PAGE_SIZE_FIELD_NUMBER: _ClassVar[int]
    DO_LAYOUT_RECOGNIZE_FIELD_NUMBER: _ClassVar[int]
    CHUNK_TOKEN_NUM_FIELD_NUMBER: _ClassVar[int]
    DELIMITER_FIELD_NUMBER: _ClassVar[int]
    filename_embd_weight: float
    task_page_size: int
    do_layout_recognize: bool
    chunk_token_num: int
    delimiter: str
    def __init__(self, filename_embd_weight: _Optional[float] = ..., task_page_size: _Optional[int] = ..., do_layout_recognize: bool = ..., chunk_token_num: _Optional[int] = ..., delimiter: _Optional[str] = ...) -> None: ...

class DocTask(_message.Message):
    __slots__ = ("file_id", "lang", "llm_id", "kb_id", "chunk_type")
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    LANG_FIELD_NUMBER: _ClassVar[int]
    LLM_ID_FIELD_NUMBER: _ClassVar[int]
    KB_ID_FIELD_NUMBER: _ClassVar[int]
    CHUNK_TYPE_FIELD_NUMBER: _ClassVar[int]
    file_id: str
    lang: str
    llm_id: str
    kb_id: str
    chunk_type: _constant_pb2.ChunkType
    def __init__(self, file_id: _Optional[str] = ..., lang: _Optional[str] = ..., llm_id: _Optional[str] = ..., kb_id: _Optional[str] = ..., chunk_type: _Optional[_Union[_constant_pb2.ChunkType, str]] = ...) -> None: ...

class Doc(_message.Message):
    __slots__ = ("doc_id", "doc_name_keyword", "title_tokens", "title_small_tokens", "page_num", "positions", "top", "content_with_weight", "content_tokens", "content_small_tokens", "image_uri", "q_vec", "q_vec_size", "image", "important_keywords", "important_keywords_tokens", "file_key", "file_md5")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    DOC_NAME_KEYWORD_FIELD_NUMBER: _ClassVar[int]
    TITLE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TITLE_SMALL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    PAGE_NUM_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    CONTENT_WITH_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_SMALL_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_URI_FIELD_NUMBER: _ClassVar[int]
    Q_VEC_FIELD_NUMBER: _ClassVar[int]
    Q_VEC_SIZE_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_KEYWORDS_TOKENS_FIELD_NUMBER: _ClassVar[int]
    FILE_KEY_FIELD_NUMBER: _ClassVar[int]
    FILE_MD5_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    doc_name_keyword: str
    title_tokens: str
    title_small_tokens: str
    page_num: _containers.RepeatedScalarFieldContainer[int]
    positions: _containers.RepeatedCompositeFieldContainer[Position]
    top: _containers.RepeatedScalarFieldContainer[int]
    content_with_weight: str
    content_tokens: str
    content_small_tokens: str
    image_uri: str
    q_vec: _containers.RepeatedScalarFieldContainer[float]
    q_vec_size: int
    image: str
    important_keywords: _containers.RepeatedScalarFieldContainer[str]
    important_keywords_tokens: str
    file_key: str
    file_md5: str
    def __init__(self, doc_id: _Optional[str] = ..., doc_name_keyword: _Optional[str] = ..., title_tokens: _Optional[str] = ..., title_small_tokens: _Optional[str] = ..., page_num: _Optional[_Iterable[int]] = ..., positions: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., top: _Optional[_Iterable[int]] = ..., content_with_weight: _Optional[str] = ..., content_tokens: _Optional[str] = ..., content_small_tokens: _Optional[str] = ..., image_uri: _Optional[str] = ..., q_vec: _Optional[_Iterable[float]] = ..., q_vec_size: _Optional[int] = ..., image: _Optional[str] = ..., important_keywords: _Optional[_Iterable[str]] = ..., important_keywords_tokens: _Optional[str] = ..., file_key: _Optional[str] = ..., file_md5: _Optional[str] = ...) -> None: ...

class Smilarity(_message.Message):
    __slots__ = ("simlarity",)
    SIMLARITY_FIELD_NUMBER: _ClassVar[int]
    simlarity: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, simlarity: _Optional[_Iterable[float]] = ...) -> None: ...

class EmbededItem(_message.Message):
    __slots__ = ("doc", "tokens")
    DOC_FIELD_NUMBER: _ClassVar[int]
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    doc: Doc
    tokens: int
    def __init__(self, doc: _Optional[_Union[Doc, _Mapping]] = ..., tokens: _Optional[int] = ...) -> None: ...

class DocItem(_message.Message):
    __slots__ = ("doc_id", "kb_ids", "doc_name_keyword", "title_tokens", "positions", "content_with_weight", "content_tokens", "image", "q_vec_size", "important_keyword", "q_vec")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    KB_IDS_FIELD_NUMBER: _ClassVar[int]
    DOC_NAME_KEYWORD_FIELD_NUMBER: _ClassVar[int]
    TITLE_TOKENS_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_WITH_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    Q_VEC_SIZE_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_KEYWORD_FIELD_NUMBER: _ClassVar[int]
    Q_VEC_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    kb_ids: _containers.RepeatedScalarFieldContainer[str]
    doc_name_keyword: str
    title_tokens: str
    positions: _containers.RepeatedCompositeFieldContainer[Position]
    content_with_weight: str
    content_tokens: str
    image: str
    q_vec_size: int
    important_keyword: _containers.RepeatedScalarFieldContainer[str]
    q_vec: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, doc_id: _Optional[str] = ..., kb_ids: _Optional[_Iterable[str]] = ..., doc_name_keyword: _Optional[str] = ..., title_tokens: _Optional[str] = ..., positions: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., content_with_weight: _Optional[str] = ..., content_tokens: _Optional[str] = ..., image: _Optional[str] = ..., q_vec_size: _Optional[int] = ..., important_keyword: _Optional[_Iterable[str]] = ..., q_vec: _Optional[_Iterable[float]] = ...) -> None: ...

class DocAggs(_message.Message):
    __slots__ = ("doc_name", "doc_id", "count")
    DOC_NAME_FIELD_NUMBER: _ClassVar[int]
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    doc_name: str
    doc_id: str
    count: int
    def __init__(self, doc_name: _Optional[str] = ..., doc_id: _Optional[str] = ..., count: _Optional[int] = ...) -> None: ...

class DocChunk(_message.Message):
    __slots__ = ("doc_id", "kb_ids", "important_keyword", "doc_name_keyword", "image", "similarity", "vector_similarity", "term_similarity", "content_with_weight", "content_tokens", "positions", "chunk_id")
    DOC_ID_FIELD_NUMBER: _ClassVar[int]
    KB_IDS_FIELD_NUMBER: _ClassVar[int]
    IMPORTANT_KEYWORD_FIELD_NUMBER: _ClassVar[int]
    DOC_NAME_KEYWORD_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    VECTOR_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    TERM_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    CONTENT_WITH_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    CONTENT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    POSITIONS_FIELD_NUMBER: _ClassVar[int]
    CHUNK_ID_FIELD_NUMBER: _ClassVar[int]
    doc_id: str
    kb_ids: _containers.RepeatedScalarFieldContainer[str]
    important_keyword: _containers.RepeatedScalarFieldContainer[str]
    doc_name_keyword: str
    image: str
    similarity: float
    vector_similarity: float
    term_similarity: float
    content_with_weight: str
    content_tokens: str
    positions: _containers.RepeatedCompositeFieldContainer[Position]
    chunk_id: str
    def __init__(self, doc_id: _Optional[str] = ..., kb_ids: _Optional[_Iterable[str]] = ..., important_keyword: _Optional[_Iterable[str]] = ..., doc_name_keyword: _Optional[str] = ..., image: _Optional[str] = ..., similarity: _Optional[float] = ..., vector_similarity: _Optional[float] = ..., term_similarity: _Optional[float] = ..., content_with_weight: _Optional[str] = ..., content_tokens: _Optional[str] = ..., positions: _Optional[_Iterable[_Union[Position, _Mapping]]] = ..., chunk_id: _Optional[str] = ...) -> None: ...

class DocDegreeProgress(_message.Message):
    __slots__ = ("message", "progress")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    message: str
    progress: float
    def __init__(self, message: _Optional[str] = ..., progress: _Optional[float] = ...) -> None: ...

class SearchRequest(_message.Message):
    __slots__ = ("kb_ids", "doc_ids", "size", "question", "vector", "topk", "similarity", "available", "page")
    KB_IDS_FIELD_NUMBER: _ClassVar[int]
    DOC_IDS_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    QUESTION_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    TOPK_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    AVAILABLE_FIELD_NUMBER: _ClassVar[int]
    PAGE_FIELD_NUMBER: _ClassVar[int]
    kb_ids: _containers.RepeatedScalarFieldContainer[str]
    doc_ids: _containers.RepeatedScalarFieldContainer[str]
    size: int
    question: str
    vector: bool
    topk: int
    similarity: float
    available: bool
    page: int
    def __init__(self, kb_ids: _Optional[_Iterable[str]] = ..., doc_ids: _Optional[_Iterable[str]] = ..., size: _Optional[int] = ..., question: _Optional[str] = ..., vector: bool = ..., topk: _Optional[int] = ..., similarity: _Optional[float] = ..., available: bool = ..., page: _Optional[int] = ...) -> None: ...

class DocSearchResult(_message.Message):
    __slots__ = ("total", "ids", "query_vector", "docs", "highlight", "aggregation", "keywords", "group_docs")
    class DocsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: DocItem
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[DocItem, _Mapping]] = ...) -> None: ...
    class HighlightEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    DOCS_FIELD_NUMBER: _ClassVar[int]
    HIGHLIGHT_FIELD_NUMBER: _ClassVar[int]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    GROUP_DOCS_FIELD_NUMBER: _ClassVar[int]
    total: int
    ids: _containers.RepeatedScalarFieldContainer[str]
    query_vector: _containers.RepeatedScalarFieldContainer[float]
    docs: _containers.MessageMap[str, DocItem]
    highlight: _containers.ScalarMap[str, str]
    aggregation: _containers.RepeatedScalarFieldContainer[str]
    keywords: _containers.RepeatedScalarFieldContainer[str]
    group_docs: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, total: _Optional[int] = ..., ids: _Optional[_Iterable[str]] = ..., query_vector: _Optional[_Iterable[float]] = ..., docs: _Optional[_Mapping[str, DocItem]] = ..., highlight: _Optional[_Mapping[str, str]] = ..., aggregation: _Optional[_Iterable[str]] = ..., keywords: _Optional[_Iterable[str]] = ..., group_docs: _Optional[_Iterable[str]] = ...) -> None: ...

class DocSearchVector(_message.Message):
    __slots__ = ("q_vec_size", "top_k", "similarity", "num_candidates", "query_vector")
    Q_VEC_SIZE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    NUM_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    QUERY_VECTOR_FIELD_NUMBER: _ClassVar[int]
    q_vec_size: int
    top_k: int
    similarity: float
    num_candidates: int
    query_vector: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, q_vec_size: _Optional[int] = ..., top_k: _Optional[int] = ..., similarity: _Optional[float] = ..., num_candidates: _Optional[int] = ..., query_vector: _Optional[_Iterable[float]] = ...) -> None: ...

class TokenizerItem(_message.Message):
    __slots__ = ("token", "synonyms", "weight", "fine_grained_tokens")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    SYNONYMS_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    FINE_GRAINED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    token: str
    synonyms: _containers.RepeatedScalarFieldContainer[str]
    weight: float
    fine_grained_tokens: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, token: _Optional[str] = ..., synonyms: _Optional[_Iterable[str]] = ..., weight: _Optional[float] = ..., fine_grained_tokens: _Optional[_Iterable[str]] = ...) -> None: ...

class TokenWeight(_message.Message):
    __slots__ = ("token", "weight")
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    token: str
    weight: float
    def __init__(self, token: _Optional[str] = ..., weight: _Optional[float] = ...) -> None: ...

class TermWeightTokens(_message.Message):
    __slots__ = ("tokens", "weight", "synonyms_tokens", "sorted_weight_tokens", "isalnum", "synonyms", "token_weights", "is_chinese")
    TOKENS_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    SYNONYMS_TOKENS_FIELD_NUMBER: _ClassVar[int]
    SORTED_WEIGHT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    ISALNUM_FIELD_NUMBER: _ClassVar[int]
    SYNONYMS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    IS_CHINESE_FIELD_NUMBER: _ClassVar[int]
    tokens: _containers.RepeatedScalarFieldContainer[str]
    weight: float
    synonyms_tokens: _containers.RepeatedScalarFieldContainer[str]
    sorted_weight_tokens: _containers.RepeatedCompositeFieldContainer[TokenizerItem]
    isalnum: bool
    synonyms: _containers.RepeatedScalarFieldContainer[str]
    token_weights: _containers.RepeatedCompositeFieldContainer[TokenWeight]
    is_chinese: bool
    def __init__(self, tokens: _Optional[_Iterable[str]] = ..., weight: _Optional[float] = ..., synonyms_tokens: _Optional[_Iterable[str]] = ..., sorted_weight_tokens: _Optional[_Iterable[_Union[TokenizerItem, _Mapping]]] = ..., isalnum: bool = ..., synonyms: _Optional[_Iterable[str]] = ..., token_weights: _Optional[_Iterable[_Union[TokenWeight, _Mapping]]] = ..., is_chinese: bool = ...) -> None: ...

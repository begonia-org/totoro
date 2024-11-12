#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/09/07 11:16:45
@Desc    :
'''
import os
import datetime
import io
import re
import time
import uuid

import magic
from snowflake import SnowflakeGenerator
import tiktoken
from urllib.parse import urlparse


gen = SnowflakeGenerator(42)

FILE_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.ms-powerpoint": "ppt",
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/zip": "zip",
    "text/html": "html",
    "application/json": "json",
    "application/xml": "xml",

    # 视频格式
    "video/mp4": "mp4",
    "video/x-msvideo": "avi",
    "video/x-matroska": "mkv",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-flv": "flv",
    "video/mpeg": "mpeg",
    "video/ogg": "ogv",

    # 音频格式
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/ogg": "ogg",
    "audio/x-wav": "wav",
    "audio/webm": "weba",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/flac": "flac",

    # 图片格式
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/webp": "webp",
    "image/tiff": "tiff",
    "image/svg+xml": "svg",
    "image/x-icon": "ico",
    "image/vnd.microsoft.icon": "ico",
    "image/vnd.wap.wbmp": "wbmp",
    "image/x-xbitmap": "xbm",
    "image/x-jg": "art",

}


def current_timestamp():
    return int(time.time() * 1000)


def timestamp_to_date(timestamp, format_string="%Y-%m-%d %H:%M:%S"):
    if not timestamp:
        timestamp = time.time()
    timestamp = int(timestamp) / 1000
    time_array = time.localtime(timestamp)
    str_date = time.strftime(format_string, time_array)
    return str_date


def date_string_to_timestamp(time_str, format_string="%Y-%m-%d %H:%M:%S"):
    time_array = time.strptime(time_str, format_string)
    time_stamp = int(time.mktime(time_array) * 1000)
    return time_stamp


def get_snowflake_id():
    snk_id = str(next(gen))
    return snk_id


def datetime_format(date_time: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(date_time.year, date_time.month, date_time.day,
                             date_time.hour, date_time.minute, date_time.second)


def get_format_time() -> datetime.datetime:
    return datetime_format(datetime.datetime.now())


def get_file_type(file) -> str:
    mime = magic.Magic(mime=True)
    if isinstance(file, str):
        mime_type = mime.from_file(file)
    else:
        if isinstance(file, io.BytesIO):
            mime_type = mime.from_buffer(file.getvalue())
            file.seek(0)
        else:
            mime_type = mime.from_buffer(file)
    return FILE_TYPES.get(mime_type, "unknown")


def is_video(file) -> str:
    ft = get_file_type(file)
    print(ft)
    if ft in ["mp4", "avi", "mkv", "webm", "mov", "flv", "mpeg", "ogv"]:
        return True
    return False


def is_audio(file) -> str:
    ft = get_file_type(file)
    if ft in ["mp3", "wav", "ogg", "weba", "m4a", "aac", "flac"]:
        return True
    return False


def rm_space(txt):
    txt = re.sub(r"([^a-z0-9.,]) +([^ ])", r"\1\2", txt, flags=re.IGNORECASE)
    return re.sub(r"([^ ]) +([^a-z0-9.,])", r"\1\2", txt, flags=re.IGNORECASE)


os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "res")
encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0


def truncate(string: str, max_len: int) -> str:
    """Returns truncated text if the length of text exceed max_len."""
    return encoder.decode(encoder.encode(string)[:max_len])


def rm_WWW(txt):
    patts = [
        (r"是*(什么样的|哪家|一下|那家|请问|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*", ""),
        (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
        (r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just|please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they|they're|you're|as|by|on|in|at|up|out|down|of) ", " ")
    ]
    for r, p in patts:
        txt = re.sub(r, p, txt, flags=re.IGNORECASE)
    return txt


def sub_special_char(line):
    return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|\+~\^])", r"\\\1", line).strip()


def is_chinese(line):
    arr = re.split(r"[ \t]+", line)
    if len(arr) <= 3:
        return True
    e = 0
    for t in arr:
        if not re.match(r"[a-zA-Z]+$", t):
            e += 1
    return e * 1. / len(arr) >= 0.7


def is_url(path):
    # 使用 urlparse 来解析路径
    parsed = urlparse(path)
    # 如果路径包含协议且网络位置（netloc）不为空，则是 URL
    return all([parsed.scheme in ["http", "https", "ftp"], parsed.netloc])


def get_now_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

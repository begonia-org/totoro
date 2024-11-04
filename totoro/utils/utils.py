#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2024/09/07 11:16:45
@Desc    :
'''
import datetime
import io
import re
import time
import uuid

import magic
from snowflake import SnowflakeGenerator

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

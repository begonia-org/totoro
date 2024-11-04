#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   image.py
@Time    :   2024/09/19 16:19:56
@Desc    :
'''
import base64
from io import BytesIO
from typing import Union

from PIL import Image


class ImageTools:
    def __init__(self) -> None:
        pass

    def image2base64(self, image: Union[Image.Image, bytes]) -> str:
        # 将Image对象转换为二进制数据
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")  # 可以根据图像类型选择适当的格式
        img_bytes = buffered.getvalue()

        # 将二进制数据编码为Base64字符串
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        return img_base64

    def base642image(self, base64_str: str) -> Image.Image:
        # 将Base64字符串解码为二进制数据
        img_bytes = base64.b64decode(base64_str)
        # 将二进制数据转换为Image对象
        image = Image.open(BytesIO(img_bytes))
        return image

#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   token.py
@Time    :   2024/09/07 15:26:26
@Desc    :
'''

import tiktoken


class ModelEncoder:
    def __init__(self) -> None:
        self.__encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def num_tokens_from_string(self, string: str) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(self.__encoder.encode(string))
        return num_tokens

    def truncate(self, string: str, max_len: int) -> int:
        """Returns truncated text if the length of text exceed max_len."""
        return self.__encoder.decode(self.__encoder.encode(string)[:max_len])
    def encode(self, string: str) -> list:
        return self.__encoder.encode(string)
    def decode(self, tokens: list) -> str:
        return self.__encoder.decode(tokens)
    

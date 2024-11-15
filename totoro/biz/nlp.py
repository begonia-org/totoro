#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nlp.py
@Time    :   2024/11/15 13:55:35
@Desc    :   
'''


import re
from typing import List, Tuple
from totoro.nlp import tokenizer, term_weight, synonym
from totoro.utils.utils import is_chinese
from totoro.pb import doc_pb2


class NLPHub:
    def __init__(self):
        self.__tokenizer = tokenizer.DocTokenizer()
        self.__term_weight = term_weight.TermWeight()
        self.__synonym = synonym.Synonym()

    def tokenize(self, content: str) -> List[str]:
        return self.__tokenizer.tokenize(content)

    def traditional2simplified(self, content: str) -> str:
        return self.__tokenizer.tradi2simp(content)

    def compute_tokens_weights(self, tokens: List[str]) -> List[Tuple[str, float]]:
        return self.__term_weight.weights(tokens)

    def fullwidth2halfwidth(self, content: str) -> str:
        return self.__tokenizer.strQ2B(content)

    def lookup_synonym(self, content: str) -> List[str]:
        return self.__synonym.lookup(content)

    def remove_redundant_words(self, txt: str) -> str:
        """移除冗余词汇

        Args:
            txt (str): _description_

        Returns:
            _type_: _description_
        """
        patts = [
            (r"是*(什么样的|哪家|一下|那家|啥样|咋样了|什么时候|何时|何地|何人|是否|是不是|多少|哪里|怎么|哪儿|怎么样|如何|哪些|是啥|啥是|啊|吗|呢|吧|咋|什么|有没有|呀)是*", ""),
            (r"(^| )(what|who|how|which|where|why)('re|'s)? ", " "),
            (r"(^| )('s|'re|is|are|were|was|do|does|did|don't|doesn't|didn't|has|have|be|there|you|me|your|my|mine|just\
             |please|may|i|should|would|wouldn't|will|won't|done|go|for|with|so|the|a|an|by|i'm|it's|he's|she's|they\
             |they're|you're|as|by|on|in|at|up|out|down) ", " ")
        ]
        for r, p in patts:
            txt = re.sub(r, p, txt, flags=re.IGNORECASE)
        return txt

    def __sub_special_char(self, line):
        return re.sub(r"([:\{\}/\[\]\-\*\"\(\)\|~\^])", r"\\\1", line).strip()

    def pre_question(self, question: str) -> List[doc_pb2.TermWeightTokens]:
        # 对输入文本进行标准化处理
        txt = re.sub(
            r"[ :\r\n\t,，。？?/`!！&\^%%]+",  # 去除特殊符号
            " ",
            self.traditional2simplified(  # 繁体转简体
                self.fullwidth2halfwidth(  # 全角转半角
                    question.lower()))).strip()  # 转小写并去除首尾空格
        txt = self.remove_redundant_words(txt)  # 去除冗余词汇
        if not is_chinese(txt):
            # 分词并计算权重
            tokens = self.tokenize(txt).split(" ")
            tokens_with_weights = self.compute_tokens_weights(tokens)
            # 清理分词中的多余字符，例如空格、双引号（"）、单引号（'）、反斜杠（\）等，确保分词在后续处理中不会因这些字符而导致错误。
            tokens_with_weights = [(re.sub(r"[ \\\"']+", "", tk), w)
                                   for tk, w in tokens_with_weights]
            return [doc_pb2.TermWeightTokens(
                is_chinese=False,  # 是否为中文
                tokens=tokens,
                token_weights=[doc_pb2.TokenWeight(token=t, weight=w) for t, w in tokens_with_weights])]
        # 中文查询逻辑部分

        def need_fine_grained_tokenize(tk):
            """
            判断是否需要对单词进行细粒度分词。
            """
            if len(tk) < 4:  # 长度小于4的词不细分
                return False
            if re.match(r"[0-9a-z\.\+#_\*-]+$", tk):  # 数字和字母等不细分
                return False
            return True
        keywords = []
        term_weight_tokens: List[doc_pb2.TermWeightTokens] = []
        # 对输入文本进行分词，最多处理256个分词结果
        for tt in self.__term_weight.split(txt)[:256]:
            if not tt:
                continue

            # 计算分词权重和同义词
            twts = self.compute_tokens_weights([tt])  # 对当前分词计算权重
            syns = self.lookup_synonym(tt)  # 查找分词的同义词
            # _tokens_with_weights = []
            sorted_token_items: List[doc_pb2.TokenizerItem] = []
            # 遍历权重词，并构造查询条件
            # 按权重降序排序
            for tk, w in sorted(twts, key=lambda x: x[1] * -1):
                # 如果需要细粒度分词，则对当前词进行更细的分词
                normalize = self.__tokenizer.fine_grained_tokenize(tk).split(
                    " ") if need_fine_grained_tokenize(tk) else []
                normalize = [
                    re.sub(
                        r"[ ,\./;'\[\]\\`~!@#$%\^&\*\(\)=\+_<>\?:\"\{\}\|，。；‘’【】、！￥……（）——《》？：“”-]+",
                        "", m) for m in normalize]  # 去除分词中的特殊字符
                normalize = [self.__sub_special_char(m)
                             for m in normalize if len(m) > 1]  # 替换特殊字符
                normalize = [m for m in normalize if len(m) > 1]  # 过滤掉长度小于2的分词
                if len(normalize) < 2:  # 如果细分结果少于2个，则不处理
                    normalize = []

                # 将当前分词加入关键词列表
                keywords.append(re.sub(r"[ \\\"']+", "", tk))

                # 构造同义词和近邻查询
                token_synonym = self.lookup_synonym(tk)  # 查找分词的同义词
                tk = self.__sub_special_char(tk)  # 替换特殊字符
                sorted_token_items.append(doc_pb2.TokenizerItem(token=tk, weight=w,
                                                                fine_grained_tokens=normalize,
                                                                synonyms=token_synonym))

            isalnum = False
            if re.match(r"[0-9a-z ]+$", tt):  # 如果分词只包含数字和字母
                isalnum = True

            # qs.append(_tokens_with_weights)  # 添加当前分词的查询条件
            term_weight_tokens.append(
                doc_pb2.TermWeightTokens(
                    is_chinese=True,
                    tokens=self.tokenize(tt),
                    synonyms_tokens=[self.__sub_special_char(
                        self.tokenize(s)) for s in syns],
                    sorted_weight_tokens=sorted_token_items,
                    isalnum=isalnum,
                    term_weight_spilt_str=tt,
                    token_weights=[doc_pb2.TokenWeight(
                        token=t, weight=w) for t, w in twts]
                )
            )
        return term_weight_tokens, keywords, [self.__tokenizer.fine_grained_tokenize(keyword) for keyword in keywords]

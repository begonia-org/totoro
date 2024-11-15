#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   term_weight.py
@Time    :   2024/11/01 09:53:48
@Desc    :   
'''


from typing import Tuple
import math
import json
import re
import os
from typing import List
import numpy as np
from totoro.nlp import tokenizer
from totoro.config import TotoroConfigure as cfg
from totoro.utils.logger import nlp_logger
from .tokenizer import DocTokenizer


class TermWeight:
    def __init__(self):
        self.tokenizer = tokenizer.DocTokenizer()
        self.stop_words = set(["请问",
                               "您",
                               "你",
                               "我",
                               "他",
                               "是",
                               "的",
                               "就",
                               "有",
                               "于",
                               "及",
                               "即",
                               "在",
                               "为",
                               "最",
                               "有",
                               "从",
                               "以",
                               "了",
                               "将",
                               "与",
                               "吗",
                               "吧",
                               "中",
                               "#",
                               "什么",
                               "怎么",
                               "哪个",
                               "哪些",
                               "啥",
                               "相关"])

        def load_dict(fnm):
            res = {}
            f = open(fnm, "r")
            while True:
                line = f.readline()
                if not line:
                    break
                arr = line.replace("\n", "").split("\t")
                if len(arr) < 2:
                    res[arr[0]] = 0
                else:
                    res[arr[0]] = int(arr[1])

            c = 0
            for _, v in res.items():
                c += v
            if c == 0:
                return set(res.keys())
            return res

        fnm = os.path.join(os.path.dirname(os.path.dirname(__file__)), "res")
        self.ne, self.df = {}, {}
        try:
            with open(os.path.join(fnm, "ner.json"), "r") as f:
                self.ne = json.load(f)
        except Exception as e:
            nlp_logger.error(f"[WARNING] Load ner.json FAIL!,{str(e)}")
        try:
            self.df = load_dict(os.path.join(fnm, "term.freq"))
        except Exception as e:
            nlp_logger.warning(f"[WARNING] Load term.freq FAIL!,{str(e)}")
        self.tokenizer = DocTokenizer()

    def pre_token(self, txt, num=False, stpwd=True):
        patt = [
            r"[~—\t @#%!<>,\.\?\":;'\{\}\[\]_=\(\)\|，。？》•●○↓《；‘’：“”【¥ 】…￥！、·（）×`&\\/「」\\]"
        ]
        rewt = [
        ]
        for p, r in rewt:
            txt = re.sub(p, r, txt)

        res = []
        for t in self.tokenizer.tokenize(txt).split(" "):
            tk = t
            if (stpwd and tk in self.stop_words) or (
                    re.match(r"[0-9]$", tk) and not num):
                continue
            for p in patt:
                if re.match(p, t):
                    tk = "#"
                    break
            tk = re.sub(r"([\+\\-])", r"\\\1", tk)
            if tk != "#" and tk:
                res.append(tk)
        return res

    def token_merge(self, tks):
        def oneTerm(t): return len(t) == 1 or re.match(r"[0-9a-z]{1,2}$", t)

        res, i = [], 0
        while i < len(tks):
            j = i
            if i == 0 and oneTerm(tks[i]) and len(
                    tks) > 1 and (len(tks[i + 1]) > 1 and not re.match(r"[0-9a-zA-Z]", tks[i + 1])):  # 多 工位
                res.append(" ".join(tks[0:2]))
                i = 2
                continue

            while j < len(
                    tks) and tks[j] and tks[j] not in self.stop_words and oneTerm(tks[j]):
                j += 1
            if j - i > 1:
                if j - i < 5:
                    res.append(" ".join(tks[i:j]))
                    i = j
                else:
                    res.append(" ".join(tks[i:i + 2]))
                    i = i + 2
            else:
                if len(tks[i]) > 0:
                    res.append(tks[i])
                i += 1
        return [t for t in res if t]

    def ner(self, t):
        if not self.ne:
            return ""
        res = self.ne.get(t, "")
        if res:
            return res

    def split(self, txt):
        """
        对输入文本进行分词，处理过程中根据特定条件合并相邻分词。

        Args:
            txt (str): 输入的文本字符串。

        Returns:
            List[str]: 返回分词后的列表，其中可能包含合并的短语。
        """

        # 初始化存储分词结果的列表
        tks = []

        # 将输入文本中的多余空格和制表符替换为单个空格，并按空格分词
        for t in re.sub(r"[ \t]+", " ", txt).split(" "):
            # 如果分词结果中已经有至少一个词，并且满足以下所有条件：
            if tks and re.match(r".*[a-zA-Z]$", tks[-1]) and re.match(r".*[a-zA-Z]$", t) and \
                  self.ne.get(t, "") != "func" and self.ne.get(tks[-1], "") != "func":

                # 合并当前分词和前一个分词，中间用空格连接
                tks[-1] = tks[-1] + " " + t
            else:
                # 如果不满足合并条件，将当前分词直接添加到分词结果中
                tks.append(t)

        # 返回最终的分词列表
        return tks

    def weights(self, tks, preprocess=True) -> List[Tuple[str, float]]:
        """
        该函数接受一个词汇列表（tks）并为每个词计算一个权重。
        权重计算过程中会涉及多个因素，取决于是否需要进行预处理（由 preprocess 参数控制）。
        如果 preprocess 为 False，则计算过程会基于 idf 和 df 计算权重。
        如果为 True，则会先对每个词进行更复杂的处理，然后再计算权重

        函数最终返回一个包含词及其权重的列表。每个词的权重经过归一化，使得所有词的权重总和为1，便于后续分析和比较。
        Args:
            tks (List[str]): 词汇列表
            preprocess (bool, optional): 是否需要预处理. Defaults to True.
        Returns:
            List[Tuple[str, float]]: 每个词及其权重

        """
        # 命名实体识别（NER）函数，给定一个词 t，返回它的权重
        def ner(t):
            # 如果词是由数字和逗号组成且长度大于2（例如：100,000），则权重为2
            if re.match(r"[0-9,.]{2,}$", t):
                return 2
            # 如果词是长度为1或2的小写字母，则权重非常低，为0.01
            if re.match(r"[a-z]{1,2}$", t):
                return 0.01
            # 如果词不在命名实体字典中，则返回默认权重1
            if not self.ne or t not in self.ne:
                return 1
            # 如果词是命名实体，返回对应类型的权重
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3, "stock": 3,
                 "firstnm": 1}
            return m[self.ne[t]]

        # 词性标注（POS tagging）函数，给定一个词 t，返回其对应的权重
        def postag(t):
            t = self.tokenizer.tag(t)  # 使用tokenizer进行词性标注
            # 如果词性标签属于副词（r）、连词（c）、助词（d），则权重为0.3
            if t in set(["r", "c", "d"]):
                return 0.3
            # 如果词性标签属于地名或专有名词（ns, nt），则权重为3
            if t in set(["ns", "nt"]):
                return 3
            # 如果是普通名词（n），则权重为2
            if t in set(["n"]):
                return 2
            # 如果是数字（匹配数字形式），则权重为2
            if re.match(r"[0-9-]+", t):
                return 2
            # 默认权重为1
            return 1

        # 计算词频函数，给定一个词 t，返回其频率的权重
        def freq(t):
            # 如果词是数字或符合特定格式（如有点、逗号等），返回权重3
            if re.match(r"[0-9. -]{2,}$", t):
                return 3
            # 获取词的实际频率
            s = self.tokenizer.freq(t)
            # 如果频率为空且词是小写字母组成的，则返回一个较高的频率权重300
            if not s and re.match(r"[a-z. -]+$", t):
                return 300
            # 如果没有返回有效频率，则设为0
            if not s:
                s = 0

            # 如果频率为0且词的长度大于等于4，则进行细粒度分词
            if not s and len(t) >= 4:
                s = [tt for tt in self.tokenizer.fine_grained_tokenize(
                    t).split(" ") if len(tt) > 1]
                if len(s) > 1:
                    # 对于多个分词，返回它们频率的最小值，降低权重
                    s = np.min([freq(tt) for tt in s]) / 6.
                else:
                    s = 0

            # 返回最大值，确保词频权重大于等于10
            return max(s, 10)

        # 计算文档频率函数（DF），给定一个词 t，返回其文档频率的权重
        def df(t):
            # 如果词是数字或符合特定模式（如有点、逗号等），返回权重5
            if re.match(r"[0-9. -]{2,}$", t):
                return 5
            # 如果词在文档频率字典中，返回字典中的值加上3
            if t in self.df:
                return self.df[t] + 3
            # 如果词是由字母组成，则返回较高的文档频率权重300
            elif re.match(r"[a-z. -]+$", t):
                return 300
            # 如果词的长度大于等于4，则进行细粒度分词，并计算文档频率
            elif len(t) >= 4:
                s = [tt for tt in self.tokenizer.fine_grained_tokenize(
                    t).split(" ") if len(tt) > 1]
                if len(s) > 1:
                    return max(3, np.min([df(tt) for tt in s]) / 6.)

            # 默认返回文档频率3
            return 3

        # 计算逆文档频率（IDF）函数，给定词频 s 和总文档数 N，返回对应的IDF值
        def idf(s, N):
            return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))

        tw = []  # 用于存储每个词及其计算出来的权重

        # 如果不需要预处理（preprocess=False）
        if not preprocess:
            # 计算每个词的IDF权重
            idf1 = np.array([idf(freq(t), 10000000) for t in tks])  # 基于词频的IDF
            idf2 = np.array([idf(df(t), 1000000000)
                            for t in tks])  # 基于文档频率的IDF
            # 计算权重，结合了命名实体识别、词性标注、频率等因素
            wts = (0.3 * idf1 + 0.7 * idf2) * \
                np.array([ner(t) * postag(t) for t in tks])
            tw = zip(tks, wts)  # 返回每个词及其权重
        else:
            # 如果需要预处理（preprocess=True）
            for tk in tks:
                # 对词进行预处理，例如合并成更有意义的子词
                tt = self.token_merge(self.pre_token(tk, True))
                # 计算每个细粒度分词的IDF权重
                idf1 = np.array([idf(freq(t), 10000000) for t in tt])
                idf2 = np.array([idf(df(t), 1000000000) for t in tt])
                # 计算权重
                wts = (0.3 * idf1 + 0.7 * idf2) * \
                    np.array([ner(t) * postag(t) for t in tt])
                # 将结果扩展到最终的tw列表
                tw.extend(zip(tt, wts))

        # 计算所有权重的总和
        S = np.sum([s for _, s in tw])

        # 返回每个词及其权重的归一化结果
        return [(t, s / S) for t, s in tw]

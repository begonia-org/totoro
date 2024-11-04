#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tokenizer.py
@Time    :   2024/11/01 09:53:40
@Desc    :   
'''


import copy
import math
import os
import re

import string
import sys
import traceback
import nltk
import datrie
from hanziconv import HanziConv
from nltk import word_tokenize
from nltk.data import find

from nltk.stem import PorterStemmer, WordNetLemmatizer

from totoro.pb import doc_pb2
from totoro.utils.image import ImageTools
# from totoro.opendoc.parser.pdf import OpenDocPdfParser
from totoro.utils.logger import nlp_logger


class DocTokenizer(object):

    def key_(self, line):
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm):
        nlp_logger.debug(f"[HUQIE]:Build trie {fnm}")
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1
            self.trie_.save(fnm + ".trie")
            of.close()
        except Exception as e:
            nlp_logger.error(
                f"[HUQIE]:Faild to build {fnm} trie, {str(e)},{traceback.format_exc()}")

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.table_regx = re.compile(
            r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>")
        self.image_tool = ImageTools()
        self.DENOMINATOR = 1000000
        self.trie_ = datrie.Trie(string.printable)
        root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        self.DIR_ = os.path.join(
            root, "res", "huqie")

        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        self.SPLIT_CHAR = r"([ ,\.<>/?;'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"
        try:
            self.trie_ = datrie.Trie.load(self.DIR_ + ".txt.trie")
        except Exception as e:
            nlp_logger.error(
                f"[HUQIE]:Build default trie error {str(e)},{traceback.format_exc()}")
            self.trie_ = datrie.Trie(string.printable)
        self._init_nltk()

        self.loadDict_(self.DIR_ + ".txt")

    def _init_nltk(self):
        # nltk.download()
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
        except Exception as e:
            print(f"[HUQIE]:Download punkt_tab error {str(e)}")
            nltk.download('punkt_tab')
        try:
            nltk.data.find("corpora/wordnet.zip")
        except Exception as e:
            import traceback
            print(f"[HUQIE]:Download wordnet error {str(e)},traceback:{traceback.format_exc()}")
            nltk.download('wordnet')

    def tokenize_table_kb(self, d: doc_pb2.Doc, t, eng):
        d.content_with_weight = t
        t = self.table_regx.sub(" ", t)
        d.content_tokens = self.tokenize(t)
        d.content_small_tokens = self.fine_grained_tokenize(d.content_tokens)

    def tokenize_chunks(self, chunks, doc: doc_pb2.Doc, eng, pdf_parser):
        res = []
        # wrap up as es documents
        for ck in chunks:
            if len(ck.strip()) == 0:
                # print(f"[HUQIE]:Empty chunk {ck}")
                continue
            # print("--", ck)
            d = copy.deepcopy(doc)
            if pdf_parser:
                try:
                    img, poss = pdf_parser.crop(ck, need_position=True)
                    d.image = self.image_tool.image2base64(img) if img else ""

                    self.add_positions(d, poss)
                    ck = pdf_parser.remove_tag(ck)
                except NotImplementedError as e:
                    nlp_logger.error(
                        f"[HUQIE]:Faild to crop image, {str(e)},{traceback.format_exc()}")
            self.tokenize_table_kb(d, ck, eng)
            res.append(d)
        return res

    def tokenize_table(self, tbls, doc: doc_pb2.Doc, eng, batch_size=10):
        res = []
        # add tables
        for (img, rows), poss in tbls:
            if not rows:
                continue
            if isinstance(rows, str):
                d = copy.deepcopy(doc)
                self.tokenize_table_kb(d, rows, eng)
                d.content_with_weight = rows
                if img:
                    d.image = self.image_tool.image2base64(img)
                if poss:
                    self.add_positions(d, poss)
                res.append(d)
                continue
            de = "; " if eng else "； "
            for i in range(0, len(rows), batch_size):
                d = copy.deepcopy(doc)
                r = de.join(rows[i:i + batch_size])
                self.tokenize_table_kb(d, r, eng)
                d.image = self.image_tool.image2base64(img)
                self.add_positions(d, poss)
                res.append(d)
        return res

    def add_positions(self, d: doc_pb2.Doc, poss):
        if not poss:
            return
        page_num = []
        position = []
        top_list = []
        for pn, left, right, top, bottom in poss:
            page_num.append(int(pn + 1))
            top_list.append(int(top))
            position.append(
                doc_pb2.Position(
                    number=int(pn + 1),
                    left=int(left),
                    right=int(right),
                    top=int(top),
                    bottom=int(bottom)))
        d.page_num.extend(page_num)
        d.positions.extend(position)
        d.top.extend(top_list)

    def loadUserDict(self, fnm):
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception as e:
            nlp_logger.warning(
                f"[HUQIE]:Load {fnm} trie error {str(e)},{traceback.format_exc()}")
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def strQ2B(self, ustring):
        """把字符串全角转半角"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # 转完之后不是半角字符返回原来的字符
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def tradi2simp(self, line):
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        # MAX_L = 10
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        F /= len(tks)
        L /= len(tks)
        if self.DEBUG:
            print("[SC]", tks, len(tks), L, F, B / len(tks) + L + F)
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split(" ")
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return [self.stemmer.stem(self.lemmatizer.lemmatize(t)) if re.match(r"[a-zA-Z_-]+$", t) else t for t in tks]

    def tokenize(self, line):
        """对输入文本进行预处理，过滤掉短文本段和特定格式的文本段。
           使用最大正向匹配和最大反向匹配对文本段进行分词。
           计算正向和反向匹配结果之间的差异，并选择得分更高的匹配结果。
           对匹配结果中连续一致和不一致的部分进行处理，生成最终的分词结果。
           使用双向最大匹配法对文本进行分词，对每个词进行词形还原和词干提取。


        Args:
            line (_type_): _description_

        Returns:
            _type_: _description_
        """
        # 输入预处理：
        # 将全角字符转换为半角字符
        line = self.strQ2B(line).lower()
        # 将繁体中文转换为简体中文
        line = self.tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            # 非中文处理，对英文文本进行分词，对每个词进行词形还原和词干提取
            return " ".join([self.stemmer.stem(self.lemmatizer.lemmatize(t)) for t in word_tokenize(line)])
        # 使用指定的分隔符对文本进行分割
        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                res.append(L)
                continue
            # print(L)

            # use maxforward for the first time
            # 对文本段进行最大正向匹配
            tks, s = self.maxForward_(L)
            # 对文本段进行最大反向匹配
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                nlp_logger.debug("[FW]", tks, s)
                nlp_logger.debug("[BW]", tks1, s1)
            # 计算正向和反向匹配结果的差异，并选择更优的匹配结果
            diff = [0 for _ in range(max(len(tks1), len(tks)))]
            for i in range(min(len(tks1), len(tks))):
                if tks[i] != tks1[i]:
                    diff[i] = 1

            if s1 > s:
                tks = tks1

            i = 0
            while i < len(tks):
                s = i
                while s < len(tks) and diff[s] == 0:
                    s += 1
                if s == len(tks):
                    res.append(" ".join(tks[i:]))
                    break
                if s > i:
                    res.append(" ".join(tks[i:s]))

                e = s
                while e < len(tks) and e - s < 5 and diff[e] == 1:
                    e += 1

                tkslist = []
                self.dfs_("".join(tks[s:e + 1]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                i = e + 1

        res = " ".join(self.english_normalize_(res))
        if self.DEBUG:
            nlp_logger.info("[TKS]", self.merge_(res))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):
        """预处理和中文比例判断。
基于长度和正则表达式的初步过滤。
通过深度优先搜索进行细粒度分词，并选择最优分词结果。
进行英文规范化处理并返回最终结果。

        Args:
            tks (_type_): _description_

        Returns:
            _type_: _description_
        """
        tks = tks.split(" ")
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    tks = []
    for t in txt.split(" "):
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


if __name__ == '__main__':
    tknzr = DocTokenizer(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    tks = tknzr.tokenize(
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    print(tknzr.fine_grained_tokenize(tks))
    # tks = tknzr.tokenize(
    #     "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    # print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("虽然我不怎么玩")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    print(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    print(tknzr.fine_grained_tokenize(tks))
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.DEBUG = False
    tknzr.loadUserDict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        print(tknzr.tokenize(line))
    of.close()

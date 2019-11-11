import pandas as pd
import numpy as np
from pandas import Series

_PAD = '<PAD>'


def gen_int_map(column: Series, sep=' ', add_pad=True, start=0):
    """
    对column列的字符串，将所有字符串分割后产生一个字典映射（单词——>数字索引）
    :param column: 处理的列
    :param sep: 分割符号
    :param add_pad: 是否添加空白处理符 <pad>
    :param start: 索引起始
    :return:
    """
    word_set = set()
    for val in column.str.split(sep):
        word_set.update(val)

    if add_pad:
        word_set.add(_PAD)

    word_int_map = {val: ii for ii, val in enumerate(word_set, start=start)}

    return word_int_map


def int_encode(word_int_map: {"word": 1}, count: int, sep=' ', pad: int = 0):
    """
    字符串分割后，转成长度为定长（count）的数字列表，如果长度小于count则用pad填充，大于count则截断
    :param word_int_map:word到数字的映射字段
    :param count: 数组定长
    :param sep: 字符串分割符
    :param pad: 填充值
    :return:
    """

    def helper(unit):
        _words = [word_int_map[word] for word in unit.split(sep)]
        if len(_words) > count:
            return np.array(unit[:count])  # 大于count则截断
        else:
            # 如果长度小于count则用pad填充
            title_vector = np.asarray([pad] * count)  # np.zeros(count)
            title_vector[:len(_words)] = _words
            return title_vector

    return helper


def multi_hot_encode(word_int_map: {"word": 1}, sep=' '):
    """
    字符串分割后，multi-hot编码
    :param word_int_map: 到数字的映射字典
    :param sep: 字符串分割符
    :return:
    """

    def helper(unit):
        word_int_list = [word_int_map[word] for word in unit.split(sep)]
        multi_hot = np.zeros(len(word_int_map))
        multi_hot[word_int_list] = 1
        return multi_hot

    return helper


def col2ints(col: Series, count=None, word_int_map=None, **kwargs):
    """
    将DF中的某一列转换为一组数字来表达
    :param col:
    :param count: 一组数字的个数
    :param word_int_map: 到数字的映射字典，默认自动生成
    :param kwargs:
    :return:
    """
    if not word_int_map:  # 默认自动生成
        word_int_map = gen_int_map(col, **kwargs)  # 映射表

    if not count:  # 默认按照映射字典的条目数
        count = len(word_int_map)

    pad = word_int_map[_PAD]
    col = col.map(int_encode(word_int_map, count=count, pad=pad, **kwargs))
    return col, word_int_map

# TODO 为完成，或不需要
# def col2int(col: Series, **kwargs):
#     """
#     将DF中的某一列转换为单个数字来表达
#     :param col:
#     :param kwargs:
#     :return:
#     """
#     word_int_map = gen_int_map(col, **kwargs)  # 映射表
#     col = col.map(int_encode(word_int_map, **kwargs))
#     return col, word_int_map

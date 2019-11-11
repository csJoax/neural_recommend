import re

import pandas as pd
from .load_csv import load_table
from utils.type import TableConfig
from .preprocess import col2ints, gen_int_map


def process_fn(items: pd.DataFrame):
    """
    处理电影的数据
    """
    # 将Title中的年份分离出来单独成列
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    titles, years = zip(
        *items.apply(lambda row: pattern.match(row['Title']).groups(), axis=1))

    items['Title'] = titles
    items['MovieYear'] = years
    # items.insert(2,'Movie Year',years)

    # 电影类型转数字字典
    items['Genres'], genres_int_dict = col2ints(items['Genres'], sep='|')

    # 电影Title转数字字典
    title_count = 15
    items['Title'], title_int_dict = \
        col2ints(items['Title'], sep=' ', count=title_count)

    return title_count, title_int_dict, genres_int_dict, items


def load_items(items_param: TableConfig, **kwargs):

    file, title, filter = items_param
    # 读取Movie数据集
    items = load_table(file, title, **kwargs)
    items_orig = items.values

    title_count, title_set, genres2int, items = process_fn(items)
    return title_count, title_set, genres2int, items, items_orig

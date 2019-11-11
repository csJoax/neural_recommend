import pandas as pd
from .load_csv import load_table
from .users import load_users
from .items import load_items
from utils.type import TableConfig


def load_ratings(ratings_param: TableConfig, **kwargs):

    file, title, filter = ratings_param

    # 读取评分数据集
    ratings = load_table(file, title, regex_filter=filter, **kwargs)
    return ratings


def load_data(users_param: TableConfig, items_param: TableConfig, ratings_param: TableConfig,
              **kwargs):  # kwargs:    sep='::', header=None,  engine='python'
    """
    从文件中加载数据集
    """
    # 读取User数据
    users, users_orig = load_users(users_param, **kwargs)

    ######################################################################################
    # 读取Movie数据集
    title_count, title_set, genres2int, items, items_orig \
        = load_items(items_param, **kwargs)

    ######################################################################################
    # 读取评分数据集
    ratings = load_ratings(ratings_param, **kwargs)

    ######################################################################################
    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), items)

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(
        target_fields, axis=1), data[target_fields]

    features = features_pd.values
    targets_values = targets_pd.values

    return title_count, title_set, genres2int, features, targets_values, \
        ratings, users, items, data, \
        items_orig, users_orig

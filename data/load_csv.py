import pandas as pd
import re


def load_table(csv_file, df_title, regex_filter=None, df_fn=None, **kwargs):
    """
    从CSV文件中读取数据，返回DF
    :param csv_file:
    :param df_title:
    :param regex_filter:
    :param kwargs:
    :return:
    """
    df = pd.read_table(csv_file, names=df_title, **kwargs)
    if regex_filter:
        df = df.filter(regex=regex_filter)

    if df_fn:
        df_fn(df)

    return df

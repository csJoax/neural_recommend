import re
import pandas as pd
from abc import ABC, abstractmethod
from utils.type import TableConfig


def load_csv(csv_file, df_title, regex_filter=None, **kwargs):
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

    return df


def load_db():
    """
    TODO 从数据库中读取数据，返回DF
    """
    # df=pd.read_sql()
    # df=pd.read_sql_query()
    pass


class ITable(ABC):
    """
    用户数据的接口
    """

    def __init__(self, param: TableConfig, **kwargs):
        file, title, filter = param
        df = load_csv(file, title, regex_filter=filter, **kwargs)
        self._orig = df.values
        self._df = self.process_fn(df)

    @abstractmethod
    def process_fn(self, df):
        pass

    @property
    def orig(self):
        return self._orig

    @property
    def df(self):
        return self._df

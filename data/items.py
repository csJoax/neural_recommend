import re
import pandas as pd
from .itable import ITable


# TODO 暂时不使用，只留下接口
class Items(ITable):

    def process_fn(self, items_df: pd.DataFrame):
        return items_df

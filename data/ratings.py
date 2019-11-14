import pandas as pd
from .itable import ITable


# TODO 暂时不使用，只留下接口
class Ratings(ITable):
    def process_fn(self, ratings_df):
        return ratings_df

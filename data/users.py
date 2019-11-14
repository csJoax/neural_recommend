from .itable import ITable


# TODO 暂时不使用，只留下接口
class Users(ITable):
    def process_fn(self, users_df):
        return users_df

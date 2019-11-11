import pandas as pd
from .load_csv import load_table
from utils.type import TableConfig


def process_user_fn(users: pd.DataFrame):
    """
    处理用户的数据
    """
    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    return users


def load_users(users_param: TableConfig, **kwargs):
    file, title, filter = users_param
    # 读取User数据
    users = load_table(file, title, regex_filter=filter, **kwargs)
    users_orig = users.values

    users = process_user_fn(users)
    return users, users_orig

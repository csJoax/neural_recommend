import re
import pandas as pd
from sklearn.model_selection import train_test_split

from .download import download_extract
from utils.type import Dataset, TableConfig
from .itable import ITable, load_csv
from .preprocess import col2ints, gen_int_map


class Users(ITable):
    def process_fn(self, users_df):
        """
        处理用户的数据
        """
        # 改变User数据中性别和年龄
        gender_map = {'F': 0, 'M': 1}
        users_df['Gender'] = users_df['Gender'].map(gender_map)

        age_map = {val: ii for ii, val in enumerate(set(users_df['Age']))}
        users_df['Age'] = users_df['Age'].map(age_map)

        return users_df


class Items(ITable):
    def __init__(self, items_param: TableConfig, title_count=15, genres_count=None, **kwargs):
        self.title_count = title_count
        self.genres_count = genres_count
        self.genres_int_dict = None
        self.title_int_dict = None
        ITable.__init__(self, items_param, **kwargs)

    def process_fn(self, items_df: pd.DataFrame):
        """
        处理电影的数据
        """
        # 将Title中的年份分离出来单独成列
        pattern = re.compile(r'^(.*)\((\d+)\)$')

        titles, years = zip(
            *items_df.apply(lambda row: pattern.match(row['Title']).groups(), axis=1))

        items_df['Title'] = titles
        # items.insert(2,'Movie Year',years)
        items_df['MovieYear'] = years

        # print(items.loc[:5,"MovieYear"]) # TODO 处理年份

        # 电影类型转数字字典
        if not self.genres_count:
            self.genres_int_dict = gen_int_map(items_df['Genres'], sep='|', add_pad=True, start=0)
            self.genres_count = len(self.genres_int_dict)
        items_df['Genres'], self.genres_int_dict = \
            col2ints(items_df['Genres'], word_int_map=self.genres_int_dict, sep='|', count=self.genres_count)

        # 电影Title转数字字典
        items_df['Title'], self.title_int_dict = col2ints(items_df['Title'], sep=' ', count=self.title_count)

        return items_df


class Ratings(ITable):
    def process_fn(self, ratings_df):
        return ratings_df


class RecommendData(object):
    def __init__(self, users: Users, items: Items, ratings: Ratings):
        self.users = users
        self.items = items
        self.ratings = ratings

        # 合并三个表
        data = pd.merge(pd.merge(ratings.df, users.df), items.df)

        # 将数据分成X和y两张表
        target_fields = ['ratings']
        features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]

        self.features = features_pd.values
        self.targets = targets_pd.values


def get_ytc():
    pass


def get_movielens(movielens=None):
    if not movielens:
        movielens = Dataset(name='ml-1m', url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                            hashcode='c4d9eecfca2ab87c1945afe126590906')
    download_extract(movielens.name, data_path='./data_dir/',
                     url=movielens.url, hash_code=movielens.hashcode)

    data_dir = './data_dir/ml-1m/'
    users = TableConfig(file=data_dir + '/users.dat',
                        title=['UserID', 'Gender', 'Age', 'JobID', 'Zip-code'],
                        filter='UserID|Gender|Age|JobID')

    items = TableConfig(file=data_dir + '/movies.dat',
                        title=['MovieID', 'Title', 'Genres'],  # TODO 加入年份处理
                        filter=None)

    ratings = TableConfig(file=data_dir + '/ratings.dat',
                          title=['UserID', 'MovieID', 'ratings', 'timestamps'],
                          filter='UserID|MovieID|ratings')

    user_table = Users(users, sep='::', header=None, engine='python')
    item_table = Items(items, sep='::', header=None, engine='python')
    rating_table = Ratings(ratings, sep='::', header=None, engine='python')

    ml_data = RecommendData(user_table, item_table, rating_table)
    return ml_data


def get_train_test(ml_data: RecommendData, test_size=0.2, batch_sizes=(256, 256), **kwargs):
    """
    获取 train、 test
    :param test_size:
    :param kwargs:
    :return:
    """
    features, targets = ml_data.features, ml_data.targets  # (features, targets)
    train_X, test_X, train_y, test_y = train_test_split(features, targets, test_size=test_size, **kwargs)

    batch_size, test_batch_size = batch_sizes  # (训练集batch_size，测试集batch_size)
    train_part = DatasetPart(train_X, train_y, "train", batch_size)
    test_part = DatasetPart(test_X, test_y, "test", test_batch_size)
    return train_part, test_part


def get_batches(Xs, ys, batch_size):
    """
    取得batch
    :param Xs:
    :param ys:
    :param batch_size:
    :return:
    """
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]


class DatasetPart(object):
    """
    数据集的一部分，是训练集、测试集、评估集的抽象，可以为 train, test, val
    暂时只当成2个部分：train, test,
    """

    def __init__(self, Xs, ys, kind, batch_size: int = 256):
        kinds = ['train', 'test', 'val']
        if kind in kinds:
            self._kind = kind
        else:
            raise ValueError("该数据集部分的命名必须取其中一种%s。" % (str(kinds)), kind)

        if isinstance(batch_size, int) and batch_size > 0:
            self._batch_size = batch_size
        else:
            raise ValueError("batch_size必须为正整数：", batch_size)

        self._example_num = min(len(Xs), len(ys))  # 样本数量
        self._batch_num = self._example_num // self._batch_size

        if len(Xs) != len(ys):
            print("Warn: 样本的数量不一致。")

        self._Xs = Xs[:self._example_num]  # X: 特征列
        self._ys = ys[:self._example_num]  # y: 标签列

    def get_batches(self):
        """
        取得batch
        :return:
        """
        for start in range(0, self._example_num, self._batch_size):
            end = min(start + self._batch_size, self._example_num)
            yield self._Xs[start:end], self._ys[start:end]

    @property
    def example_num(self):
        return self._example_num

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def batch_num(self):
        return self._batch_num

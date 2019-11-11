from .download import download_extract
from .ratings import load_data
from utils.type import Dataset, TableConfig


movielens = Dataset(name='ml-1m', url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
                    hashcode='c4d9eecfca2ab87c1945afe126590906')

data_dir = './data_dir/ml-1m/'
users = TableConfig(file=data_dir+'/users.dat',
                    title=['UserID', 'Gender', 'Age', 'JobID', 'Zip-code'],
                    filter='UserID|Gender|Age|JobID')

items = TableConfig(file=data_dir+'/movies.dat',
                    title=['MovieID', 'Title', 'Genres'],
                    filter=None)

ratings = TableConfig(file=data_dir+'/ratings.dat',
                      title=['UserID', 'MovieID', 'ratings', 'timestamps'],
                      filter='UserID|MovieID|ratings')

# name : (url, hash_code)
data_dict = {
    'ml-1m': ('http://files.grouplens.org/datasets/movielens/ml-1m.zip', 'c4d9eecfca2ab87c1945afe126590906')
}


def get_movielens():
    download_extract(movielens.name, data_path='./data_dir/',
                     url=movielens.url, hash_code=movielens.hashcode)

    all_data = load_data(users,items,ratings, sep='::', header=None,  engine='python')

    #  ratings, users, items
    return  all_data


def get_ytc():
    pass

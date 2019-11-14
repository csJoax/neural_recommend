from config.nn_param import *
from config.file_param import *
from models.tf1.tcn import TextCNN
from data.get_data import get_movielens

param_dict = {
    'title_int_num': title_int_num,
    'categ_int_num':categ_int_num,
    'embed_dim': embed_dim,
    'num_outputs': num_outputs,

    'filter_num': filter_num,
    'window_sizes': window_sizes,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'show_every_n_batches': show_every_n_batches,
    'dropout_keep': dropout_keep,
    'save_dir': save_dir,
    'load_dir': save_dir
}


def train_model():
    data = get_movielens()
    model = TextCNN(data, param_dict)
    model.build()

    model.train(num_epochs)

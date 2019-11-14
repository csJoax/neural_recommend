import pickle

import numpy as np
import tensorflow.compat.v1 as tf  # tf version: 2.0

# from config.movielens_param import *
# from config.nn_param import *
from .units import Units, InputUnits, EmbedUnits, FCUnits, TextConvUnits, Combination
from .summary import get_summaries
from .logger import LossLogger
from data.get_data import get_batches, get_train_test, RecommendData

from sklearn.model_selection import train_test_split


# todo


# TODO 剥离 Movielens 业务逻辑
class TextCNN(object):
    """
    用于推荐系统的文本卷积网络
    """

    def __init__(self, data: RecommendData, param_dict: dict):
        for k in param_dict:
            if isinstance(k, str):
                setattr(self, k, param_dict[k])
        # units
        self._inputs = None
        self._user_embeds = None
        self._item_embeds = None
        self._user_combined_feature = None
        self._item_combined_feature = None

        # tf
        self._train_graph = None
        self._inference = None

        self._cost = None
        self._loss = None

        self._global_step = None
        self._optimizer = None
        self._gradients = None
        self._train_op = None

        # DFs
        self._data = data
        self._users = data.users
        self._items = data.items

        features = data.features
        self.user_id_max = max(features.take(0, 1)) + 1  # 6040 # 用户ID个数
        self.gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2 # 性别个数
        self.age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7 # 年龄类别个数
        self.job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21 # 职业个数
        self.movie_id_max = max(features.take(1, 1)) + 1  # 3952  # 电影ID个数
        self.movie_categories_max = max(data.items.genres_int_dict.values()) + 1  # 18 + 1 = 19    # 电影类型个数
        self.movie_title_max = len(data.items.title_int_dict)  # 5216   # 电影名单词个数

        self._losses = {'train': [], 'test': []}

    def get_inputs(self):
        """
        获取输入单元
        :return:
        """
        inputs = InputUnits()
        inputs.update(["user_id", "user_gender", "user_age", "user_job", "movie_id"], dtype=tf.int32, shape=[None, 1])
        inputs.add("movie_categories", shape=[None, self.categ_int_num])
        inputs.add("movie_titles", shape=[None, self.title_int_num])

        inputs.add("targets", shape=[None, 1])
        inputs.update(["LearningRate", "dropout_keep_prob"], shape=None, dtype=tf.float32)
        return inputs

    def get_user_embeds(self, inputs: InputUnits):
        """
        User的嵌入矩阵
        :param inputs:
        :return:
        """
        embed_dim = self.embed_dim
        user_embeds = EmbedUnits("user_embedding")
        user_embeds.add("user_id", inputs, [self.user_id_max, embed_dim])
        user_embeds.add("user_gender", inputs, [self.gender_max, embed_dim // 2])
        user_embeds.add("user_age", inputs, [self.age_max, embed_dim // 2])
        user_embeds.add("user_job", inputs, [self.job_max, embed_dim // 2])
        return user_embeds

    def get_item_embeds(self, inputs: InputUnits):
        """
        item的嵌入矩阵
        :param inputs:
        :return:
        """
        embed_dim = self.embed_dim
        item_embeds = EmbedUnits("movie_embedding")
        item_embeds.add("movie_id", inputs, [self.movie_id_max, embed_dim])  # todo item_id_max
        item_embeds.add("movie_categories", inputs, [self.movie_categories_max, embed_dim], operation='sum')
        item_embeds.add("movie_titles", inputs, [self.movie_title_max, embed_dim], operation='expand')
        return item_embeds

    # 将User的嵌入矩阵一起全连接生成User的特征#
    def get_user_combined_feature(self, inputs: EmbedUnits, fc_outputs):
        """
        User的组合特征
        :param inputs:
        :param fc_outputs:
        :return:
        """
        user_fcs = FCUnits("user_fc", self.embed_dim)
        user_fcs.update(["user_id", "user_gender", "user_age", "user_job"],
                        layer_input=inputs, activation=tf.nn.relu)
        user_combine = Combination("user", user_fcs)
        user_combine.combine(axis=2, fc_outputs=fc_outputs, shape_flat=[-1, fc_outputs], activation=tf.tanh)
        return user_combine

    # Movie Title的文本卷积网络实现#
    def get_item_convs(self, inputs: Units, dropout_keep_prob):
        item_convs = TextConvUnits()
        item_convs.add("movie_titles", inputs, self.embed_dim, self.filter_num, self.title_int_num,
                       window_sizes=self.window_sizes, dropout_keep_prob=dropout_keep_prob)
        return item_convs

    # todo 将Movie的各个层一起做全连接#
    def get_movie_combined_feature(self, inputs, conv, fc_outputs):
        """
        item的组合特征
        :param inputs:
        :param conv:
        :param fc_outputs:
        :return:
        """
        movie_fcs = FCUnits("movie_fc", self.embed_dim)
        movie_fcs.update(["movie_id", "movie_categories"], layer_input=inputs)
        movie_combine = Combination("item", movie_fcs)
        movie_combine.append(conv)
        movie_combine.combine(axis=2, fc_outputs=fc_outputs, activation=tf.tanh)
        return movie_combine

    def build(self):
        """
        构建计算图
        :return:
        """
        tf.reset_default_graph()
        self._train_graph = train_graph = tf.Graph()
        with train_graph.as_default():
            # 获取输入占位符
            self._inputs = inputs = self.get_inputs()
            # 获取User的4个嵌入向量
            self._user_embeds = user_embed = self.get_user_embeds(inputs)
            # 得到用户特征
            self._user_combined_feature = user_combine = \
                self.get_user_combined_feature(user_embed, fc_outputs=self.num_outputs)
            user_combine_layer, user_combine_layer_flat = user_combine.get_combine_layer()

            # 获取电影ID的嵌入向量
            self._item_embeds = movie_embed = self.get_item_embeds(inputs)

            # 获取电影名的特征向量
            item_convs = self.get_item_convs(inputs=movie_embed["movie_titles"],
                                             dropout_keep_prob=inputs["dropout_keep_prob"])
            pool_layer_flat, dropout_layer = item_convs["movie_titles"]

            # 得到电影特征
            self._item_combined_feature = movie_combine = \
                self.get_movie_combined_feature(movie_embed, dropout_layer, fc_outputs=self.num_outputs)
            movie_combine_layer, movie_combine_layer_flat = movie_combine.get_combine_layer()

            # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
            with tf.name_scope("inference"):
                # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
                #         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
                #         inference = tf.layers.dense(inference_layer, 1,
                #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")
                # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
                #        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat))
                inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat, axis=1)
                self._inference = tf.expand_dims(inference, axis=1)

            with tf.name_scope("loss"):
                # MSE损失，将计算值回归到评分
                self._cost = tf.losses.mean_squared_error(inputs['targets'], self._inference)
                self._loss = tf.reduce_mean(self._cost)
            # 优化损失
            #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
            self._global_step = tf.Variable(0, name="global_step", trainable=False)
            self._optimizer = tf.train.AdamOptimizer(inputs['LearningRate'])
            self._gradients = self._optimizer.compute_gradients(self._loss)  # cost
            self._train_op = self._optimizer.apply_gradients(self._gradients, global_step=self._global_step)

    # TODO 剥离 Movielens 业务逻辑
    def get_feed(self, batches, dropout_keep):
        """
        获取输入层的输入映射
        :param batches: 批的迭代器
        :param dropout_keep:
        :return:
        """

        inputs = self._inputs
        batch_size = self.batch_size

        x, y = next(batches)

        categories = np.zeros([batch_size, self.categ_int_num])
        for i in range(batch_size):
            categories[i] = x.take(6, 1)[i]

        titles = np.zeros([batch_size, self.title_int_num])
        for i in range(batch_size):
            titles[i] = x.take(5, 1)[i]

        feed = {
            inputs['user_id']: np.reshape(x.take(0, 1), [batch_size, 1]),
            inputs['user_gender']: np.reshape(x.take(2, 1), [batch_size, 1]),
            inputs['user_age']: np.reshape(x.take(3, 1), [batch_size, 1]),
            inputs['user_job']: np.reshape(x.take(4, 1), [batch_size, 1]),
            inputs['movie_id']: np.reshape(x.take(1, 1), [batch_size, 1]),
            inputs['movie_categories']: categories,  # x.take(6,1)
            inputs['movie_titles']: titles,  # x.take(5,1)
            inputs['targets']: np.reshape(y, [batch_size, 1]),
            inputs['dropout_keep_prob']: dropout_keep,  # dropout_keep
            inputs['LearningRate']: self.learning_rate
        }
        return feed

    def train(self, num_epochs):
        """
        训练网络
        :return:
        """

        self._losses = {'train': [], 'test': []}

        with tf.Session(graph=self._train_graph) as sess:
            # 搜集数据给tensorBoard用
            train_summary_op, train_summary_writer, \
            inference_summary_op, inference_summary_writer \
                = get_summaries(sess, self._gradients, self._loss)

            sess.run(tf.global_variables_initializer())
            for epoch_i in range(num_epochs):

                # 将数据集分成训练集和测试集，随机种子不固定
                train_set, test_set = get_train_test(self._data, batch_sizes=(256, 256))

                train_batches = train_set.get_batches()
                test_batches = test_set.get_batches()

                train_logger = LossLogger(train_set, self.show_every_n_batches)
                test_logger = LossLogger(train_set, self.show_every_n_batches)

                # 训练的迭代，保存训练损失
                for batch_i in range(train_set.batch_num):
                    feed = self.get_feed(train_batches, self.dropout_keep)

                    step, train_loss, summaries, _ = sess.run(
                        [self._global_step, self._loss, train_summary_op, self._train_op], feed)  # cost

                    self._losses['train'].append(train_loss)  # 保存训练损失
                    train_summary_writer.add_summary(summaries, step)  #
                    train_logger.check_print(epoch_i, batch_i, train_loss)

                # 使用测试数据的迭代
                for batch_i in range(test_set.batch_num):
                    feed = self.get_feed(test_batches, dropout_keep=1)

                    step, test_loss, summaries = sess.run(
                        [self._global_step, self._loss, inference_summary_op], feed)  # cost

                    self._losses['test'].append(test_loss)  # 保存测试损失
                    inference_summary_writer.add_summary(summaries, step)  #
                    test_logger.check_print(epoch_i, batch_i, test_loss)

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, self.save_dir+"textCNN")  # , global_step=epoch_i
            print('Model Trained and Saved')

    def test(self):
        pass

    def get_tensor(self, name):
        return self._train_graph.get_tensor_by_name(name + ":0")

    # TODO load
    def rating_item(self, user_id_val, movie_id_val):
        movies = self._items.df
        users = self._users.df

        # 电影ID转下标的字典，数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
        movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}

        loaded_graph = tf.Graph()  #
        with tf.Session(graph=loaded_graph) as sess:  #
            # Load saved model
            loader = tf.train.import_meta_graph(self.load_dir + '.meta')
            loader.restore(sess, self.load_dir)

            categories = np.zeros([1, 18])
            categories[0] = movies.values[movieid2idx[movie_id_val]][2]

            titles = np.zeros([1, self.title_int_num])
            titles[0] = movies.values[movieid2idx[movie_id_val]][1]

            feed = {
                self.get_tensor('user_id'): np.reshape(users.values[user_id_val - 1][0], [1, 1]),
                self.get_tensor('user_gender'): np.reshape(users.values[user_id_val - 1][1], [1, 1]),
                self.get_tensor('user_age'): np.reshape(users.values[user_id_val - 1][2], [1, 1]),
                self.get_tensor('user_job'): np.reshape(users.values[user_id_val - 1][3], [1, 1]),
                self.get_tensor('movie_id'): np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
                self.get_tensor('movie_categories'): categories,  # x.take(6,1)
                self.get_tensor('movie_titles'): titles,  # x.take(5,1)
                self.get_tensor('dropout_keep_prob'): 1
            }

            # Get Prediction
            inference_val = sess.run([self.get_tensor('inference')], feed)

            return (inference_val)

    def save_params(params, path='data_dir/params.p'):
        """
        保存参数到文件中
        """
        pickle.dump(params, open(path, 'wb'))

    def load_params(path='data_dir/params.p'):
        """
        从文件中加载参数
        """
        return pickle.load(open(path, mode='rb'))

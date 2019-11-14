from typing import Union, Iterable
from abc import ABC, abstractmethod
import tensorflow.compat.v1 as tf  # tf version: 2.0


class Units(ABC):
    """
    神经网络中的一些单元
    """

    def __init__(self):
        self.units_dict = {}

    @abstractmethod
    def create_unit(self, name, layer_input, *args, **kwargs):
        pass

    def add(self, name, layer_input=None, *args, **kwargs):
        """
        添加一个单元
        """
        if name in self.units_dict:
            print("Warn: 该嵌入单元已经存在，将被覆盖。")

        if isinstance(layer_input, Units):
            if name in layer_input:
                layer_input = layer_input[name]
            else:
                raise KeyError(name + "相关的值不在数字的单元组中", layer_input)

        self.create_unit(name, layer_input, *args, **kwargs)

    def update(self, name: Union[str, Iterable[str]], *args, **kwargs):
        """
        添加多个单元
        :param name: 单元名称，可以是多个名字的数组
        :return:
        """
        if isinstance(name, str):
            self.add(name, *args, **kwargs)
        elif isinstance(name, Iterable):
            for n in name:
                self.add(n, *args, **kwargs)
        else:
            raise TypeError("name必须为字符串或者字符串数组")

    def __getitem__(self, name):
        return self.units_dict[name]

    def __contains__(self, item):
        return item in self.units_dict

    def units(self):
        return self.units_dict.values()


# class Merger(Units):
class Combination(object):
    """
    神经网络中的的单元合并
    """

    def __init__(self, name, branches: Union[list, Units] = None):
        if not branches:
            branches = []
        elif isinstance(branches, FCUnits):
            branches = list(branches.units())

        self._branches = branches
        self._combine_layer = None
        self._combine_layer_flat = None
        self._use_fc = True
        self._name = name
        self._name_scope = name + "_combine"

    def append(self, branch):
        self._branches.append(branch)

    @property
    def name(self):
        return self._name

    def combine(self, axis, shape_flat=None,
                force=False, use_fc=True, fc_outputs=200,
                **kwargs):  # kwargs:  activation_fn=tf.tanh,
        """
        创建所有全连接单元的连接组合，并返回该组合层。如果连接组合已经创建，直接返回该组合
        :param axis: 合并多个输入时的轴
        :param shape_flat: reshape时输出的形状，一般会扁平化
        :param force: 强行合并，即使 连接组合 已经存在，也创建（新的覆盖旧的）
        :param use_fc: 是否使用全连接
        :param fc_outputs: 使用全连接时的输出单元数量
        :param kwargs: 全连接的其他参数
        :return:
        """

        if force or (not self._combine_layer or not self._combine_layer_flat):
            with tf.name_scope(self._name_scope):
                self._combine_layer = tf.concat(self._branches, axis=axis, name=self.name + "_concat")
                self._use_fc = use_fc
                if use_fc:  # 根据具体情况决定是否使用全连接
                    self._combine_layer = tf.layers.dense(self._combine_layer, fc_outputs,
                                                          name=self.name + "_combine_fc",
                                                          **kwargs)  # (?, 1, 200)
                if not shape_flat:
                    shape_flat = [-1, fc_outputs]
                self._combine_layer_flat = tf.reshape(self._combine_layer, shape_flat,
                                                      name=self.name + "_combine_flat")
        return self.get_combine_layer()

    def get_combine_layer(self):
        return self._combine_layer, self._combine_layer_flat


class InputUnits(Units):
    """
    神经网络的输入占位符
    """

    def create_unit(self, name, layer_input=None, dtype=tf.int32, shape=(None, 1), *args, **kwargs):
        """
        创建一个输入占位符
        :param name: 输入占位符名称
        :param layer_input: 输入层无需 layer_input
        :param dtype: 占位符数据类型
        :param shape: 占位符的形状
        :return:
        """
        self.units_dict[name] = tf.placeholder(dtype=dtype, shape=shape, name=name)


class EmbedUnits(Units):
    """
    神经网络的嵌入单元
    """

    def __init__(self, name_scope: str):
        """
        :param name_scope: 这些嵌入单元所属的命名空间
        """
        Units.__init__(self)
        self._name_scope = name_scope

    def create_unit(self, name, layer_input, shape=None, minval=-1, maxval=1, operation=None, *args, **kwargs):
        """
        创建一个嵌入单元
        :param name:  嵌入单元的名称
        :param layer_input: 嵌入单元的输入
        :param shape: 嵌入单元的形状
        :param minval:
        :param maxval:
        :param operation: 对多个嵌入向量的组合处理，默认为None，可取 "sum", "expand"
        :return:
        """
        if not shape:
            raise ValueError("shape不可为空。")

        with tf.name_scope(self._name_scope):
            embed_matrix = tf.Variable(tf.random_uniform(shape, minval, maxval), name=name + "_embed_matrix")
            embed_layer = tf.nn.embedding_lookup(embed_matrix, layer_input, name=name + "_embed_layer")
            if operation == "sum":
                embed_layer = tf.reduce_sum(embed_layer, axis=1, keep_dims=True)
            elif operation == "expand":
                embed_layer = tf.expand_dims(embed_layer, -1)

            self.units_dict[name] = embed_layer


class FCUnits(Units):
    """
    神经网络的全连接单元
    """

    def __init__(self, name_scope: str, input_dim):
        """
        :param name_scope: 这些全连接单元所属的命名空间
        """
        Units.__init__(self)
        self._name_scope = name_scope
        self._input_dim = input_dim
        self._combine_layer = None
        self._combine_layer_flat = None

    def create_unit(self, name, layer_input, activation=tf.nn.relu, *args, **kwargs):
        """
        创建一个全连接单元
        :param name:  全连接单元的名称
        :param layer_input: 全连接单元的输入
        :param activation: 全连接单元的激活函数
        :return:
        """
        with tf.name_scope(self._name_scope):
            fc_layer = tf.layers.dense(layer_input, self._input_dim, name=name + "_fc_layer", activation=activation)
            self.units_dict[name] = fc_layer


class TextConvUnits(Units):
    """
    神经网络的文本卷积单元
    """

    def create_unit(self, name, layer_input, embed_dim=0, filter_num=64, sentences_size=0,
                    window_sizes=(2, 3, 4, 5),
                    dropout_keep_prob=None, *args, **kwargs):
        """
        添加一个文本卷积单元
        :param name:  文本卷积单元的名称
        :param layer_input: 文本卷积单元的输入
        :param embed_dim:
        :param filter_num: 文本卷积核数量
        :param sentences_size: 电影名长度 # 15
        :param window_sizes: 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
        :param dropout_keep_prob:
        :return:
        """

        # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
        pool_layer_lst = []
        for window_size in window_sizes:
            with tf.name_scope(name + "_conv_maxpool_{}".format(window_size)):
                filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                             name="filter_weights")
                filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")

                conv_layer = tf.nn.conv2d(layer_input, filter_weights, [1, 1, 1, 1], padding="VALID", name="conv_layer")
                relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias), name="relu_layer")

                maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                               padding="VALID", name="maxpool_layer")
                pool_layer_lst.append(maxpool_layer)

        # 聚合
        with tf.name_scope(name + "_concat"):
            pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")
            max_num = len(window_sizes) * filter_num
            pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")
            self.units_dict[name + "_pool_layer_flat"] = pool_layer_flat

        # Dropout层
        with tf.name_scope(name + "_pool_dropout"):
            dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")
            self.units_dict[name + "_dropout_layer"] = dropout_layer

    def __getitem__(self, name):
        return self.units_dict[name + "_pool_layer_flat"], \
               self.units_dict[name + "_dropout_layer"]

    def __contains__(self, name):
        name += "_pool_layer_flat"
        return name in self.units_dict

import datetime
from data.get_data import DatasetPart

class LossLogger(object):
    """
    负责输出训练时记录和打印每个batch的loss
    """

    def __init__(self, data_part:DatasetPart, show_every_n_batches):
        """
        :param example_num: 样本数量
        :param batch_size: 批尺寸大小
        """
        self._example_num = data_part.example_num
        self._batch_size = data_part.batch_size
        self._batch_num = data_part.batch_num  # 批数量
        self._show_every_n_batches = show_every_n_batches  # 每隔多少个批次打印一次损失函数值

    def print_loss(self, epoch_i, batch_i, loss):
        """
        打印损失
        :param epoch_i:
        :param batch_i:
        :param loss:
        :return:
        """
        time_str = datetime.datetime.now().isoformat()

        print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
            time_str, epoch_i, batch_i, self._batch_num, loss)
        )

    def check_print(self, epoch_i, batch_i, loss: float):
        """
        检查是否到达打印周期
        :param epoch_i:
        :param batch_i:
        :param loss:
        :return:
        """
        if (epoch_i * self._batch_num + batch_i) % self._show_every_n_batches == 0:
            self.print_loss(epoch_i, batch_i, loss)

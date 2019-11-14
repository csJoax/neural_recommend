# 网络模型相关的参数


# 电影名长度
title_int_num = title_count = 15
# 电影类型长度
categ_int_num = 19
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = (2, 3, 4, 5)
# 文本卷积核数量
filter_num = 8

# Number of Epochs
num_epochs = 5

# Batch Size
batch_size = 256
# todo test_batch_size = 256
# todo batch_sizes = (batch_size, test_batch_size)

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.001
# Show stats for every n number of batches
show_every_n_batches = 20

num_outputs = 200

# 嵌入矩阵的维度
embed_dim = 32

# 对电影类型嵌入向量做加和操作的标志，考虑过使用mean做平均，但是没实现mean
combiner = "sum"

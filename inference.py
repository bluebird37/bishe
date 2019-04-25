# -*- coding:utf-8 -*-
import tensorflow as tf
import statistical
import parameterset
size1=parameterset.size1
size2=parameterset.size2
# 定义CNN网络结构的参数
INPUT_NODE = size1*size2*parameterset.NUM_CHANNELS  # 输入层节点
OUTPUT_NODE = statistical.personname_number  # 输出层节点

IMAGE_SIZE1 = size1
IMAGE_SIZE2 = size2
NUM_CHANNELS = parameterset.NUM_CHANNELS
NUM_LABELS = OUTPUT_NODE

# C1６个　C3十个

# 第一层卷积层的尺度和深度
CONV1_DEEP = 6
CONV1_SIZE = 5

# 第二层卷积层的尺度和深度
CONV2_DEEP = 10
CONV2_SIZE = 5

# 全连接层输出
FC_SIZE = 128


# CNN网络结构
def inference(input_tensor, train, regularizer):
    # 第一层卷积层
    with tf.variable_scope('layer1-conv1'):#在 tf.name_scope下时，tf.get_variable()创建的变量名
        # 不受 name_scope 的影响，而且在未指定共享变量时，
        # 如果重名会报错，tf.Variable()会自动检测有没有变量重名，如果有则会自行处理。
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv1_biases = tf.get_variable(
            "bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0)
        )
        conv1 = tf.nn.conv2d(
            input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.avg_pool(
            relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )

    # 第三层卷积层
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable(
            'weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        conv2_biases = tf.get_variable(
            'bias', [CONV2_DEEP],
            initializer=tf.constant_initializer(0.0)
        )
        # 使用边长为5，深度为10的过滤器，过滤器移动的步长为1，且使用全0填充
        conv2 = tf.nn.conv2d(
            pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'
        )
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层池化层
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.avg_pool(
            relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'
        )
        # 将第四层的输出转化为向量
        pool_shape = pool2.get_shape().as_list()
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层，全连接层，使用dropout函数避免过拟合问题
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable(#创建变量
            'weight', [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1)#初始化器，stddev为标准差
            #生成维度为【nodes,FC_SIZE】的标准差为0.1的随机数
        )
        # 在全连接层加入正则化
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))#把变量放入一个集合，把很多变量变成一个列表
            #losses为key，value为list，这里为regularizer
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE], initializer=tf.constant_initializer(0.1)
        )
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)
    # 第六层，全连接层
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1)
        )
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias', [NUM_LABELS],
            initializer=tf.constant_initializer(0.1)
        )
        logit = tf.add(tf.matmul(fc1, fc2_weights), fc2_biases)

    return logit

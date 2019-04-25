# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import inference
import statistical
import parameterset
size1=parameterset.size1
size2=parameterset.size2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
SIZE1 = size1
SIZE2 = size2
test_all=statistical.personname_number
train_all=statistical.train_all
# 配置CNN参数
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = parameterset.TRAINING_STEPS
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
LEARNING_RATE_BASE = parameterset.LEARNING_RATE_BASE  # 基础的学习率，对于orl数据库的40个人的400张图片训练，学习率为0.1时的准确率为95%，
# 对于自己拍照的一个人的20张图片训练，学习率设置为0.01，准确率为90%
LEARNING_RATE_DECAY = 0.99  # 学习的衰减率
BATCH_SIZE = parameterset.BATCH_SIZE
MODEL_SAVE_PATH = "./path/to/model/"
MODEL_NAME = "model.ckpt"

def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(statistical.personname_number)  # 返回来一个给定形状和类型的用0填充的数组；
        tmp[label[i] - 1] = 1
        ys.append(tmp)
    return ys


def train(data, label):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE, SIZE1, SIZE2, parameterset.NUM_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, inference.OUTPUT_NODE], name='y-output')

    # 使用L2正则化计算损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * BATCH_SIZE
    image_batch, label_batch = tf.train.shuffle_batch(
        [data, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue
    )

    y = inference.inference(x, False, regularizer)

    global_step = tf.Variable(0, trainable=False)  # 创建变量
    # tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)
    # 参数名称	       参数类型	   含义
    # initial_          value	 所有可以转换为Tensor的类型	变量的初始值
    # trainable	        bool	如果为True，会把它加入到GraphKeys.TRAINABLE_VARIABLES，才能对它使用Optimizer
    # collections	    list	指定该图变量的类型、默认为[GraphKeys.GLOBAL_VARIABLES]
    # validate_shape	bool	如果为False，则不进行类型和维度检查
    # name	           string	变量的名称，如果没有指定则系统会自动分配一个唯一的值
    # 第一个参数initial_value是必需的

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step
    )
    # tf.train.ExponentialMovingAverage(decay, steps)
    # 这个函数用于更新参数，就是采用滑动平均的方法更新参数。
    # 这个函数初始化需要提供一个衰减速率（decay），用于控制模型的更新速度。
    # 这个函数还会维护一个影子变量（也就是更新参数后的参数值），这个影子变量的初始值就是这个变量的初始值，
    # 影子变量值的更新方式如下：
    # shadow_variable = decay * shadow_variable + (1-decay) * variable
    # shadow_variable是影子变量，variable表示待更新的变量，也就是变量被赋予的值，
    # decay为衰减速率。decay一般设为接近于1的数（0.99,0.999）。decay越大模型越稳定，因为decay越大，
    # 参数更新的速度就越慢，趋于稳定。
    # tf.train.ExponentialMovingAverage这个函数还提供了自己动更新decay的计算方式：
    # decay= min（decay，（1+steps）/（10+steps））
    # steps是迭代的次数，可以自己设定。

    # 滑动平均更新变量的操作
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算交叉熵作为刻画预测值和真实值之间的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    #tf.argmax(y_, 1),
    # 一般来说，对于分类或者回归模型进行评估时，需要使得模型在训练数据上似的损失函数值最小，即使得经验风险函数(Empirical
    # risk)最小化，但是如果只考虑经验风险，容易出现过拟合，因此还需要考虑模型的泛化性，一般常用的方法就是在目标函数中加上正则项，有损失项（loss
    # term）加上正则项（regularization
    # term）构成结构风险（Structural
    # risk）。
    # cross_entropy交叉熵
    # 交叉熵刻画的是两个概率分布之间的距离，是分类问题中使用比较广泛的损失函数之一。给定两个概率分布p和q，通过交叉熵计算的两个概率分布之间的距离为：
    #                               H(X=x)=−∑xp(x)logq(x)
    # 我们通过softmax回归将神经网络前向传播得到的结果变成交叉熵要求的概率分布得分。
    # logits: 神经网络的最后一层输出，如果有batch的话，它的大小为[batch_size, num_classes], 单样本的话大小就是num_classes
    # labels: 样本的实际标签，大小与logits相同。且必须采用labels=y_，logits=y的形式将参数传入。
    # 具体的执行流程大概分为两步，第一步首先是对网络最后一层的输出做一个softmax，这一步通常是求取输出属于某一类的概率，对于单样本而言，
    # 就是输出一个num_classes大小的向量[Y1, Y2, Y3, ....], 其中Y1, Y2, Y3分别表示属于该类别的概率，
    # softmax的公式为：softmax(x)i = exp(xi)∑jexp(xj)
    # 第二步是对softmax输出的向量[Y1, Y2, Y3, ...]，和样本的时机标签做一个交叉熵，公式如下：
    #             Hy′(y) =−∑iy′ilog(yi)
    # 其中y′i指代实际标签向量中的第i个值，yi就是softmax的输出向量[Y1, Y2, Y3, ...]中的第i个元素的值。
    # 显而易见。预测yi越准确，结果的值就越小（前面有负号），最后求一个平均，就得到我们想要的loss了
    # 这里需要注意的是，这个函数返回值不是一个数，而是一个向量，如果要求交叉熵，我们要在做一步tf.resuce_sum操作，
    # 就是对向量里面的所有元素求和, 最后就能得到Hy′(y), 如果要求loss，则需要做一步tf.reduce_mean操作，对向量求均值.

    # 计算所有样例中交叉熵的平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_all / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True
    )
    # 优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate) \
        .minimize(loss, global_step=global_step)
    # tf.train.GradientDescentOptimizer(learning_rate)作用：创建一个梯度下降优化器对象
    # 参数：
    # learning_rate: A Tensor or a floating point value. 要使用的学习率
    # use_locking: 要是True的话，就对于更新操作（update operations.）使用锁
    # name: 名字，可选，默认是”GradientDescent”.
    # tf.train.Optimizer.minimize参数（1）loss：即最小化的目标变量，一般就是训练的目标函数，均方差或者交叉熵；
    # （2）global_step：梯度下降一次加1，一般用于记录迭代优化的次数，主要用于参数输出和保存；
    # （3）var_list 每次要迭代更新的参数集合。
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()

    # 验证
    # accuracy = tf.reduce_mean()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)#启动入队线程，由多个或单个线程，
        # 按照设定规则，把文件读入Filename Queue中。函数返回线程ID的列表，
        # 一般情况下，系统有多少个核，就会启动多少个入队线程（入队具体使用多少个线程在tf.train.batch中定义）
        # 迭代的训练网络
        for i in range(TRAINING_STEPS):
            xs, ys = sess.run([image_batch, label_batch])
            xs = xs / 255.0
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                          SIZE1,
                                          SIZE2,
                                          parameterset.NUM_CHANNELS))
            # 将图像和标签数据通过tf.train.shuffle_batch整理成训练时需要的batch
            ys = get_label(ys)
            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: reshaped_xs, y_: ys})

            if i % 100 == 0:
                # 每100轮输出一次在训练集上的测试结果
                acc = loss.eval({x: reshaped_xs, y_: ys})
                print("After %d training step[s], loss on training"
                      " batch is %g. " % (step, loss_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )
            if i == TRAINING_STEPS:
                # 保存最后一次训练结果
                acc = loss.eval({x: reshaped_xs, y_: ys})
                print("After %d training step[s], loss on training"
                      " batch is %g. " % (step, loss_value))

                saver.save(
                    sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step
                )
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    # 显示tfrecord格式的图片
    filename_queue = tf.train.string_input_producer(["train.tfrecords"])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [SIZE1, SIZE2, parameterset.NUM_CHANNELS])
    label = tf.cast(features['label'], tf.int32)
    train(img, label)


if __name__ == '__main__':
    tf.app.run()

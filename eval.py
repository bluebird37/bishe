# -*- coding:utf-8 -*-
import time
import tensorflow as tf
import numpy as np
import inference
import train
import cv2
import os
import parameterset
import statistical

size1 = parameterset.size1
size2 = parameterset.size2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


def get_label(label):
    ys = []
    for i in range(label.size):
        tmp = np.zeros(statistical.personname_number)
        tmp[label[i] - 1] = 1
        ys.append(tmp)
    return ys


def evaluate():  # //评估
    with tf.Graph().as_default() as g:
        filename_queue = tf.train.string_input_producer(["test.tfrecords"])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string),
                                           })
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [size1, size2, parameterset.NUM_CHANNELS])
        label = tf.cast(features['label'], tf.int32)
        min_after_dequeue = 100
        capacity = min_after_dequeue + 3 * 200
        image_batch, label_batch = tf.train.shuffle_batch(
            [img, label], batch_size=statistical.test_all,
            capacity=capacity, min_after_dequeue=min_after_dequeue
        )

        x = tf.placeholder(tf.float32,
                           [statistical.test_all,
                            inference.IMAGE_SIZE1,
                            inference.IMAGE_SIZE2,
                            parameterset.NUM_CHANNELS],
                           name='x-input')
        y_ = tf.placeholder(
            tf.float32, [None, inference.OUTPUT_NODE], name='y-input'
        )

        y = inference.inference(x, None, None)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(
            train.MOVING_AVERAGE_DECAY
        )
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)
        # 每隔EVAL_INTERVAL_SECS秒调用一次
        while True:
            with tf.Session() as sess:
                #test = cv2.imread('./test_scene/徐钰卓/1.jpg')
                test = cv2.imdecode(np.fromfile('./test_scene/徐钰卓/1.jpg', dtype=np.uint8), -1)
                #test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
                test = np.array(test)
                test = test / 255.0
                test_re = np.reshape(test, (1, size1, size2, parameterset.NUM_CHANNELS))

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                # TensorFlow提供了两个类来实现对Session中多线程的管理：tf.Coordinator和
                # tf.QueueRunner，这两个类往往一起使用。
                # Coordinator类用来管理在Session中的多个线程，可以用来同时停止多个工作线程并且向那个在等待所有工作线程终止的程序报告异常，该线程捕获到这个异常之后就会终止所有线程。使用
                # tf.train.Coordinator()
                # 来创建一个线程管理器（协调器）对象。
                # QueueRunner类用来启动tensor的入队线程，可以用来启动多个工作线程同时将多个tensor（训练数据）推送入文件名称队列中，具体执行函数是
                # tf.train.start_queue_runners ， 只有调用tf.train.start_queue_runners
                # 之后，才会真正把tensor推入内存序列中，供计算单元调用，否则会由于内存序列为空，数据流图会处于一直等待状态
                xs, ys = sess.run([image_batch, label_batch])
                ys = get_label(ys)
                xs = xs / 255.0
                validate_feed = {x: xs,
                                 y_: ys}

                cpkt = tf.train.get_checkpoint_state(
                    train.MODEL_SAVE_PATH
                )
                # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名。
                # tf.train.get_checkpoint_state(checkpoint_dir, latest_filename=None)
                # 该函数返回的是checkpoint文件CheckpointState
                # proto类型的内容，其中有model_checkpoint_path和all_model_checkpoint_paths两个属性。其中
                # model_checkpoint_path保存了最新的tensorflow模型文件的文件名，
                # all_model_checkpoint_paths则有未被删除的所有tensorflow模型文件的文件名。
                if cpkt and cpkt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, cpkt.model_checkpoint_path)
                    # 通过文件名得到模型保存时迭代的轮数
                    global_step = cpkt.model_checkpoint_path \
                        .split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training steps, validation "
                          "accuracy = %.4f" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
def main(argv=None):
    evaluate()


if __name__ == '__main__':
    tf.app.run()


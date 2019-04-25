# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import statistical
# 显示tfrecord格式的图片
path = "./orl_scene/"
personname_number = statistical.personname_number
i = 0
train_path = "./train_scene/"
test_path = "./test_scene/"
####
classes =statistical.classes
writer_train = tf.python_io.TFRecordWriter("train.tfrecords")
writer_test = tf.python_io.TFRecordWriter("test.tfrecords")

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串类型
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def generate():
    # 遍历字典
    for index, name in classes.items():
        train = train_path + str(name) + '/'
        test = test_path + str(name) + '/'
        for img_name in os.listdir(train):
            img_path = train + img_name  # 每一个图片的地址
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index + 1),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer_train.write(example.SerializeToString())
            print(img_path + "写入到文件：" + "train.tfrecords, 成功！")
        for img_name in os.listdir(test):
            img_path = test + img_name  # 每一个图片的地址
            # img = cv2.imread(img_path)
            img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_raw = img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index + 1),
                'img_raw': _bytes_feature(img_raw)
            }))
            writer_test.write(example.SerializeToString())
            print(img_path + "写入到文件：" + "test.tfrecords, 成功！")
    writer_test.close()
    writer_train.close()
generate()

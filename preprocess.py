# -*- coding:utf-8 -*-
import tensorflow as tf
import os
import cv2
import numpy as np
import random
import parameterset

input_path = "./orl_scene/"
train_path = "./train_scene/"
test_path = "./test_scene/"
path = "./orl_scene/"
input_path_length = len(input_path)
low=2#固定每个文件夹的测试样例为两张

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)

# 生成对应的子文件夹，现场识别时，只需新建对应训练者的文件夹
for fn in os.listdir(path):  # fn 表示的是文件名,生成与orl_scene里子文件夹对应的train_scene和
    # test_scene里对应的子文件夹
    if not os.path.exists(train_path + '/' + fn):
        os.mkdir(train_path + '/' + fn)
    if not os.path.exists(test_path + '/' + fn):
        os.mkdir(test_path + '/' + fn)
# 生成训练和测试的数据
def generate_data(train_path, test_path):
    # 生成训练和测试文件夹，用于测试识别率，现场识别时此文件夹已有，可不用此功能
    for (root, dirs, filenames) in os.walk(path):
        for dir in dirs:
            temp_path = root + dir
            index = 1
            for (sonroot, filename, filenames) in os.walk(temp_path):
                random.shuffle(filenames)
                for filename in filenames:
                    if filename.endswith('.jpg'):
                        img_path = temp_path + '/' + filename
                        folrdername = os.path.dirname(img_path)[input_path_length:]
                        # 使用opencv 读取图片
                        img_data = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                        # 按照论文中的将图片大小调整为64 * 64
                        #img_data = cv2.resize(img_data, (size, size), interpolation=cv2.INTER_AREA)
                        if index <= low:
                            #cv2.imwrite(test_path + '/' + folrdername + '/' + str(index) + '.jpg', img_data)
                            outtest_path=test_path + folrdername + '/' + str(index) + '.jpg'
                            cv2.imencode('.jpg', img_data)[1].tofile(outtest_path)
                            print("导入："+outtest_path+"成功！")
                            index += 1
                        elif index > low:
                            #cv2.imwrite(train_path + '/' + folrdername + '/' + str(index) + '.jpg', img_data)
                            outtrain_path=train_path + folrdername + '/' + str(index) + '.jpg'
                            cv2.imencode('.jpg', img_data)[1].tofile(outtrain_path)
                            print("导入：" + outtrain_path + "成功！")
                            index += 1

# def generateall_data(train_path):
#     # 生成训练和测试文件夹，用于测试识别率，现场识别时此文件夹已有，可不用此功能
#     index = 1
#     output_index = 1
#     for (dirpath, dirnames, filenames) in os.walk(input_path):
#         # 打乱文件列表，相当于是随机选取8张训练集，2张测试
#         random.shuffle(filenames)
#         for filename in filenames:
#             if filename.endswith('.jpg'):
#                 img_path = dirpath + '/' + filename
#                 folrdername = os.path.dirname(img_path)[input_path_length:]
#                 # 使用opencv 读取图片
#                 img_data = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
#                 # 按照论文中的将图片大小调整为64 * 64
#                 #img_data = cv2.resize(img_data, (size, size), interpolation=cv2.INTER_AREA)
#                 #cv2.imwrite(train_path + '/' + folrdername + '/' + str(index) + '.jpg', img_data)
#                 cv2.imencode('.jpg', img_data)[1].tofile(train_path + '/' + folrdername + '/' + str(index) + '.jpg')
#                 index += 1
#                 if index > title:
#                     output_index += 1
#                     index = 1
#generateall_data(train_path, test_path)#将所有文件作为训练样本
generate_data(train_path, test_path)#将所有文件分为训练样本和测试样本

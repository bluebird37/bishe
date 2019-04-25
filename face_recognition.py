#!/usr/bin/python
#-*- coding:utf-8 -*-
from numpy import unicode
from skimage import io, transform
import os
import tensorflow as tf
import cv2
import dlib
import inference
import tkinter.filedialog
import random
from PIL import Image, ImageDraw, ImageFont
import train
from PyQt5.QtWidgets import QMessageBox
import numpy as np
import parameterset
import statistical

size1 = parameterset.size1
size2 = parameterset.size2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
face_personname_number = 0
path = "./orl_scene/"
w = size1
h = size2
c = parameterset.NUM_CHANNELS

def relight(img, light=1, bias=0):
    wei = img.shape[1]
    hei = img.shape[0]
    # image = []
    for i in range(0, wei):
        for j in range(0, hei):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img
detector = dlib.get_frontal_face_detector()  # 获取人脸分类器
ID=statistical.ID

x = tf.placeholder(tf.float32, shape=[1, w, h, c], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, 1], name='y_')

logits = inference.inference(x, None, None)
predict = tf.argmax(logits, 1)

saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, './path/to/model/model.ckpt-' + str(parameterset.TRAINING_STEPS))  # 加载以训练好的模型
def picture_recognition(img,file_path,output_file_path):
    # 从指定路径识别图片
    ##提取人脸
    if img is not None:
        # 转为灰度图片
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)
        print("Number of faces detected: {}".format(len(dets)))
        if len(dets) == 0:
            return "图像不清晰，无法检测到人脸"
        else:
            for i, d in enumerate(dets):
                x1 = d.top() if d.top() > 0 else 0
                y1 = d.bottom() if d.bottom() > 0 else 0
                x2 = d.left() if d.left() > 0 else 0
                y2 = d.right() if d.right() > 0 else 0
                face = img[x1:y1, x2:y2]
                # 调整图片的对比度与亮度，对比度与亮度值都取随机数，这样能增加样本的多样性
                # face = relight(face, random.uniform(0.75, 1.25), random.randint(-40, 40))
                face = cv2.resize(face, (size1, size2))
                later_img = transform.resize(face, (w, h, c))
                res = sess.run(predict, feed_dict={x: [later_img]})
                print(ID[res[0]])
                ##画框
                sourceFileName = output_file_path
                #out_img = cv2.imdecode(np.fromfile(output_file_path, dtype=np.uint8), -1)#读
                cv2.rectangle(img, (x2, x1), (y2, y1), (0, 255, 0), 4)#写
                #cv2.imwrite(sourceFileName, out_img)  # 保存
                ##将识别结果标记在图片上,并显示
                # cv2.putText(图像, 文字, (x, y), 字体, 大小, (b, g, r), 宽度)
                fontSize=20
                out_img = cv2.putText(img, ID[res[0]], (x2, x1 - fontSize * 1),
                                      cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 3)#写
                if sourceFileName[len(sourceFileName)-3:]=="jpg":
                    cv2.imencode('.jpg', out_img)[1].tofile(sourceFileName)
                else:
                    cv2.imwrite(sourceFileName,out_img)
            return ID[res[0]]

    else:
        face_re_tag=0
        return 0

# def face_recognition():
#     user = input("选择图片识别（A）还是摄像头实时识别（B）:")
#     # 从指定路径识别图片
#     saver = tf.train.Saver()
#     sess = tf.Session()
#     saver.restore(sess, './path/to/model/model.ckpt-' + str(train.TRAINING_STEPS))  # 加载以训练好的模型
#     if user == "A" or user == 'a':
#         ###########
#         root = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
#         root.withdraw()  # 将Tkinter.Tk()实例隐藏
#         default_dir = "C:/Users/27117/Desktop"
#         file_path = tkinter.filedialog.askopenfilename(title='Choose File',
#                                                            initialdir=(os.path.expanduser(default_dir)))
#         root.destroy()
#         print(file_path)
#         img = cv2.imread(file_path)
#         ##提取人脸
#         if img is not None:
#             # 转为灰度图片
#             gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             # 使用detector进行人脸检测
#             dets = detector(gray_img, 1)
#             print("Number of faces detected: {}".format(len(dets)))
#             if len(dets) == 0:
#                 # 标记
#                 print("Worring：该图片无法检测到人脸")
#             else:
#                 for i, d in enumerate(dets):
#                     x1 = d.top() if d.top() > 0 else 0
#                     y1 = d.bottom() if d.bottom() > 0 else 0
#                     x2 = d.left() if d.left() > 0 else 0
#                     y2 = d.right() if d.right() > 0 else 0
#                     face = img[x1:y1, x2:y2]
#                     # 调整图片的对比度与亮度，对比度与亮度值都取随机数，这样能增加样本的多样性
#                     #face = relight(face, random.uniform(0.75, 1.25), random.randint(-40, 40))
#                     face = cv2.resize(face, (size, size))
#                     later_img = transform.resize(face, (w, h, c))
#                     res = sess.run(predict, feed_dict={x: [later_img]})
#                     #accuary=sess.run(less,feed_dict={x: [later_img]})
#                     #print(accuary)
#                     print(ID[res[0]])
#                     ##画框
#                     cv2.rectangle(img, (x2, x1), (y2, y1), (0, 255, 0), 4)
#                     cv2.imwrite(file_path, img)  # 保存
#                     ##将识别结果标记在图片上,并显示
#                     sourceFileName = file_path
#                     avatar = Image.open(sourceFileName)
#                     drawAvatar = ImageDraw.Draw(avatar)
#                     xSize, ySize = avatar.size
#                     fontSize = min(xSize, ySize) // 20
#                     myFont = ImageFont.truetype("./weiruanLight.ttc", fontSize)
#                     drawAvatar.text([x2 - fontSize * 1, x1 - fontSize * 2], \
#                                     ID[res[0]], fill=(255, 255, 0), font=myFont)
#                     del drawAvatar
#                     # avatar.save(sourceFileName)
#                     avatar.show()
#         else:
#             tkinter.messagebox.showerror('错误','没有选择文件')
#     # 打开摄像头实时识别
#     elif user == 'B' or user == 'b':
#         # 打开摄像头识别
#         cap = cv2.VideoCapture(0)
#         # 视屏封装格式
#         while True:
#             ret, frame = cap.read()
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             cv2.imshow("Press ESC to Exit", frame)
#             # 抓取图像
#             if cv2.waitKey(10) & 0xFF == ord('s'):
#                 cv2.imwrite(r'./camera_image/now.jpg', frame)
#                 camera_img = cv2.imread("./camera_image/now.jpg")
#                 camera_dets = detector(camera_img, 1)
#                 print("Number of faces detected: {}".format(len(camera_dets)))
#                 if len(camera_dets) == 0:
#                     # 标记
#                     print("光线太暗，没有检测到人脸")
#                 else:
#                     for index, face in enumerate(camera_dets):
#                         print('face {}; left {}; top {}; right {}; bottom {}'.format
#                               (index, face.left(), face.top(), face.right(), face.bottom())
#                               )
#                         left = face.left()
#                         top = face.top()
#                         right = face.right()
#                         bottom = face.bottom()
#                         camera_later_img = camera_img[top:bottom, left:right]
#                         camera_later_img = relight(camera_later_img, random.uniform(0.75, 1.25),
#                                                    random.randint(-40, 40))
#                         camera_later_img = transform.resize(camera_later_img, (w, h, c))
#                         camera_res = sess.run(predict, feed_dict={x: [camera_later_img]})
#                         print(logits)
#                         print(ID[camera_res[0]])
#             # 按esc退出
#             key = cv2.waitKey(10) & 0xff
#             if key == 27:
#                 break
#         cap.release()
#         cv2.destroyAllWindows()
#     else:
#         print("操作无效，请选择'A'or'B'")
#
# def main(argv=None):
#     # 显示tfrecord格式的图片
#     face_recognition()
#
#
# if __name__ == '__main__':
#     tf.app.run()

import os
import cv2
import numpy as np
import dlib
import parameterset
import random
size1=parameterset.size1
size2=parameterset.size2
sourse_path = "F:/bishe/ds_master_test_project/ds_master/orl_student_all/"
goal_path="./orl_scene/"
input_path_length = len(sourse_path)
detector = dlib.get_frontal_face_detector()
title=40
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

###########
for fn in os.listdir(sourse_path):  # fn 表示的是文件名,生成与orl_scene里子文件夹对应的train_scene和
    # test_scene里对应的子文件夹
    if not os.path.exists(goal_path + '/' + fn):
        os.mkdir(goal_path + '/' + fn)
############
for (root, dirs, filenames) in os.walk(sourse_path):
    for dir in dirs:
        temp_path=root+dir
        index = 1
        for (sonroot, filename, filenames) in os.walk(temp_path):
            for filename in filenames:
                if index>title:
                    break
                if filename.endswith('.jpg'):
                    img_path = sonroot + '/' + filename
                    folrdername = os.path.dirname(img_path)[input_path_length:]
                    # 使用opencv 读取图片
                    img_data = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
                    #img_data = cv2.imread(img_path)
                    # 按照论文中的将图片大小调整为size* size
                    gray_img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
                    # 使用detector进行人脸检测
                    dets = detector(gray_img, 1)
                    #print(filename)
                    if len(dets)>0:
                        for i, d in enumerate(dets):
                            x1 = d.top() if d.top() > 0 else 0
                            y1 = d.bottom() if d.bottom() > 0 else 0
                            x2 = d.left() if d.left() > 0 else 0
                            y2 = d.right() if d.right() > 0 else 0
                            face = img_data[x1:y1, x2:y2]
                            # 调整图片的对比度与亮度，对比度与亮度值都取随机数，这样能增加样本的多样性
                            #face = relight(face, random.uniform(0.75, 1.25), random.randint(-40, 40))
                            face = cv2.resize(face, (size1, size2))
                            cv2.imencode('.jpg', face)[1].tofile(goal_path + '/' + folrdername + '/' + str(index) + '.jpg')
                            #cv2.imwrite(goal_path + '/' + folrdername + '/' + str(index) + '.jpg', face)
                            print("foldername:{};     index:{}".format(folrdername,index))
                            index += 1

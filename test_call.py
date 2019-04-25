import os
import cv2
import time

sourcePath = "C:/Users/27117/Desktop/zhaoliyin"
targetPath1 = "F:/bishe/ds_master_test_project/ds_master/orl_scene/zhaoliyin"

def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray


def readPicSaveFace_1(sourcePath, targetPath, *suffix):
    try:
        ImagePaths = getAllPath(sourcePath, *suffix)

        # 对list中图片逐一进行检查,找出其中的人脸然后写到目标文件夹下
        count = 1
        # haarcascade_frontalface_alt.xml为库训练好的分类器文件，下载opencv，安装目录中可找到
        face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')
        for imagePath in ImagePaths:
            try:
                img = cv2.imread(imagePath)

                if type(img) != str:
                    faces = face_cascade.detectMultiScale(img, 1.1, 5)
                    if len(faces):
                        for (x, y, w, h) in faces:
                            # 设置人脸宽度大于16像素，去除较小的人脸
                            if w >= 64 and h >= 64:
                                # 以时间戳和读取的排序作为文件名称
                                listStr = [str(int(time.time())), str(count)]
                                fileName = ''.join(listStr)
                                # 扩大图片，可根据坐标调整
                                X = int(x)
                                W = min(int(x + w), img.shape[1])
                                Y = int(y)
                                H = min(int(y + h), img.shape[0])

                                f = cv2.resize(img[Y:H, X:W], (W - X, H - Y))
                                cv2.imwrite(targetPath + os.sep + '%s.jpg' % fileName, f)
                                count += 1
                                print(imagePath + "have face")
            except:
                continue
    except IOError:
        print("Error")
    else:
        print('Find ' + str(count - 1) + ' faces to Destination ' + targetPath)

readPicSaveFace_1(sourcePath, targetPath1, '.jpg', '.JPG', 'png', 'PNG')
import cv2
import dlib
import os
import sys
import random
import time
output_dir = './orl_student_all/goujiacheng'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index <= 40):
        print('Being processed picture %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()
        cv2.imshow("yui",img)
        cv2.imwrite(output_dir+'/'+str(index)+'.jpg', img)
        time.sleep(0.5)
        index += 1
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break
    else:
        print('Finished!')
        break

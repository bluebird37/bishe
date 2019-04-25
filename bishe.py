#!/usr/bin/python
#-*- coding:cp936 -*-
import threading
from multiprocessing import Process,Queue,Pool
import numpy as np
import multiprocessing

from PyQt5.QtCore import pyqtSignal

from bisheGui import Ui_MainWindow
import tkinter.filedialog
from PyQt5 import QtWidgets,QtGui
import os
import cv2
import tkinter.filedialog
from PyQt5.QtWidgets import QMessageBox
import face_recognition
from PyQt5 import QtWidgets, QtCore
import sys
import datetime,time

####---------------------------------
class MyPyQT_Form(QtWidgets.QMainWindow,Ui_MainWindow):
    #page2变量设置
    ##----导入图片识别
    file_path = ""
    output_file_path = ""
    #tag = 0#标志
    ##-----选择文件夹批量识别
    PictureInfolder_path=""#选择图片输入文件夹路径
    PictureOufolder_path=""#选择图片输出文件夹路径
    answer=""
    ##-----摄像头实时识别
    camera=""
    camera_number=0
    def __init__(self,qtag,qdata,qdataresu,qclose,lock):
        self.qtag=qtag
        self.qdata = qdata
        self.qdataresu=qdataresu
        self.qclose = qclose
        self.lock=lock
        super(MyPyQT_Form,self).__init__()
        self.setupUi(self)
        self.stackedWidget.setCurrentIndex(0)
        ########
        self.actiontianjia.triggered.connect(self.onAddTriggered)
        self.actiontupianshibie.triggered.connect(self.onPictureReTriggered)
        self.actionshexi.triggered.connect(self.onCaremaReTriggered)
        self.actionyan.triggered.connect(self.onMessageTriggered)
        self.actioncans.triggered.connect(self.onSetTriggered)
        #####page1-----buttons事件
        self.pushButton.clicked.connect(self.pushButtonAddclicked)#选择新添待识别用户文件夹事件
        self.pushButton_2.clicked.connect(self.pushButtonTrainclicked)#训练事件
        self.pushButton_3.clicked.connect(self.pushButtonPutclicked)#提交事件
        #####page2-----buttons事件----图片识别
        self.pushButton_5.clicked.connect(self.pushButtonPictureFileclicked)  # 选择图片事件
        self.pushButton_6.clicked.connect(self.pushButtonPictureFileReclicked)  # 选择图片识别输出事件
        self.pushButton_7.clicked.connect(self.pushButtonPictureInfolderReclicked)  # 选择图片输入文件夹
        self.pushButton_8.clicked.connect(self.pushButtonPictureOufolderReclicked)  # 选择图片输出文件夹
        self.pushButton_9.clicked.connect(self.pushButtonPicturefolderReclicked)  # 选择图片文件夹批量识别
        #######page2-----buttons事件----摄像头识别
        self.pushButton_11.clicked.connect(self.pushButtonopencameraReclicked)  # 选择图片输出文件夹
        self.pushButton_12.clicked.connect(self.pushButtonclosecameraReclicked)  # 选择图片文件夹批量识别
    #实现pushButton_click()函数，textEdit是我们放上去的文本框的id
    def onAddTriggered(self):
        self.stackedWidget.setCurrentIndex(0)
    def onPictureReTriggered(self):
        self.stackedWidget.setCurrentIndex(1)
    def onCaremaReTriggered(self):
        self.stackedWidget.setCurrentIndex(2)
    def onMessageTriggered(self):
        self.stackedWidget.setCurrentIndex(3)
    def onSetTriggered(self):
        self.stackedWidget.setCurrentIndex(4)
    ######page1-----buttons事件
    def pushButtonAddclicked(self):
        root = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
        root.withdraw()  # 将Tkinter.Tk()实例隐藏
        default_dir = "C:/Users/27117/Desktop"
        folder_path = tkinter.filedialog.askdirectory(parent=root,
                                         title="选择文件夹",
                                         initialdir=(os.path.expanduser(default_dir)))
        root.destroy()
        self.textEdit.setPlainText(folder_path)
    def pushButtonTrainclicked(self):
        print("train")
    def pushButtonPutclicked(self):
        print("put")
    #######page2-----buttons事件----图片识别
    def pushButtonPictureFileclicked(self):
        root2 = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
        root2.withdraw()  # 将Tkinter.Tk()实例隐藏
        default_dir = "C:/Users/27117/Desktop"
        my_filetypes = [('jpg files', '.jpg')]
        self.file_path = tkinter.filedialog.askopenfilename(title='选择文件',
                                                       initialdir=(os.path.expanduser(default_dir)),
                                                       filetypes=my_filetypes)
        root2.destroy()
        self.textEdit_6.setPlainText(self.file_path)
        if len(self.textEdit_6.toPlainText())>0:
            if self.file_path[len(self.file_path)-3:]=="jpg":
                self.im=cv2.imdecode(np.fromfile(self.file_path,dtype=np.uint8),-1)
                if self.im is not None:
                    self.im= cv2.resize(self.im, (450, 329))
                    cv2.imwrite('./input_lingshi_picture/temp.png', self.im)
                    #self.tag=1
                    png = QtGui.QPixmap('./input_lingshi_picture/temp.png')
                    self.label_6.setPixmap(png)
                    self.label_6.setScaledContents (True)
                else:
                    QMessageBox.warning(self, '警告', '文件格式有误，请重新选择文件')
            else:
                QMessageBox.warning(self, '警告', '没有选择以jpg结尾的文件')
        else:
            self.label_6.setPixmap(QtGui.QPixmap())

    def thread1(self, qtag, qdataresu):
        if qtag.get(True) == "reok":
            self.answer = qdataresu.get(True)
            if self.answer != "图像不清晰，无法检测到人脸":
                png2 = QtGui.QPixmap('./output_lingshi_picture/temp1.png')
                self.label_7.setPixmap(png2)
                self.label_7.setScaledContents(True)
                #self.tag = 0
                self.pushButton_6.setText("识别")
            else:
                self.pushButton_6.setText("识别")
                QMessageBox.warning(self, '警告', '图像不清晰,请选择其他图片')
    def thread2(self, qtag, qdata,qdataresu,lock):
        for (root, dirs, filenames) in os.walk(self.PictureInfolder_path):
            for filename in filenames:
                if filename[len(filename) - 3:] != "jpg":
                    self.textEdit_10.append("文件：" + filename + "不是jpg文件")
                else:
                    self.file_path = self.PictureInfolder_path + "/" + filename
                    self.output_file_path = self.PictureOufolder_path + "/" + filename
                    floder_im = cv2.imdecode(np.fromfile(self.file_path, dtype=np.uint8), -1)
                    if floder_im is not None:
                        lock.acquire()
                        qtag.put("ok")
                        qdata.put(self.file_path)
                        qdata.put(self.output_file_path)
                        lock.release()
                        if qtag.get(True) == "reok":
                            self.answer = qdataresu.get(True)
                            answer_textEdit_10_set = "文件：" + filename + "   识别结果为：" + self.answer
                            self.textEdit_10.append(answer_textEdit_10_set)
                    else:
                        self.textEdit_10.append("文件：" + filename + "   格式有误")
        self.textEdit_10.append("end!")
        self.textEdit_10.append("\n" + "详情可查看文件夹：" + self.PictureOufolder_path + "/   里的输出图片")
        self.pushButton_9.setText("识别")
    def pushButtonPictureFileReclicked(self):
        if self.pushButton_6.text()=="识别":
            if len(self.textEdit_6.toPlainText())>0:
                self.pushButton_6.setText("识别中...")
                self.file_path=self.textEdit_6.toPlainText()
                self.output_file_path='./output_lingshi_picture/temp1.png'
                ####--------两个子线程
                self.lock.acquire()
                self.qtag.put("ok")
                self.qdata.put(self.file_path)
                self.qdata.put(self.output_file_path)
                self.lock.release()
                th =  threading.Thread(target=self.thread1,args=(self.qtag,self.qdataresu))
                th.start()
            else:
                QMessageBox.warning(self, '警告', '没有选择文件')
    def pushButtonPictureInfolderReclicked(self):
        root3 = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
        root3.withdraw()  # 将Tkinter.Tk()实例隐藏
        default_dir = "C:/Users/27117/Desktop"
        self.PictureInfolder_path = tkinter.filedialog.askdirectory(parent=root3,
                                                      title="选择文件夹",
                                                      initialdir=(os.path.expanduser(default_dir)))
        root3.destroy()
        self.textEdit_8.setPlainText(self.PictureInfolder_path)
    def pushButtonPictureOufolderReclicked(self):
        root4 = tkinter.Tk()  # 创建一个Tkinter.Tk()实例
        root4.withdraw()  # 将Tkinter.Tk()实例隐藏
        default_dir = "C:/Users/27117/Desktop"
        self.PictureOufolder_path = tkinter.filedialog.askdirectory(parent=root4,
                                                               title="选择文件夹",
                                                               initialdir=(os.path.expanduser(default_dir)))
        root4.destroy()
        self.textEdit_9.setPlainText(self.PictureOufolder_path)
    def pushButtonPicturefolderReclicked(self):
        if self.pushButton_9.text()=="识别":
            self.textEdit_10.clear()
            self.textEdit_10.append("文件夹："+self.PictureInfolder_path+"/："+"\n")
            if len(self.textEdit_8.toPlainText())>0 and len(self.textEdit_9.toPlainText())>0:
                self.pushButton_9.setText("识别中...")
                th2 = threading.Thread(target=self.thread2, args=(self.qtag, self.qdata,self.qdataresu,self.lock))
                th2.start()
            elif len(self.textEdit_8.toPlainText())==0:
                QMessageBox.warning(self, '警告', '没有选择输入文件夹')
            elif len(self.textEdit_9.toPlainText())==0:
                QMessageBox.warning(self, '警告', '没有选择输出文件夹')
    #######page2-----buttons事件----摄像头识别
    def thread3(self, qtag, qdata,qdataresu,lock):
        f = open("./record/record.txt", 'a+')
        self.camera_number = 0
        cap = cv2.VideoCapture(self.camera_number + cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
        starttime = datetime.datetime.now()
        while True:
            if self.camera=="":
                ret, frame = cap.read()
                cv2.imwrite(r'./camera_image/now.jpg', frame)
                cv2.imwrite(r'./camera_image/now.png', frame)
                camerapng = QtGui.QPixmap('./camera_image/now.png')
                self.label_8.setPixmap(camerapng)
                self.label_8.setScaledContents(True)
                self.label_12.setText("摄像头已开启")
                endtime = datetime.datetime.now()
                if (endtime - starttime).seconds==3:
                    starttime=endtime
                    resulttime=time.localtime(time.time())
                    lock.acquire()
                    qtag.put("ok")
                    qdata.put("./camera_image/now.jpg")
                    qdata.put("./camera_output_image/"+str(time.strftime('%Y-%m-%d_%H-%M-%S',resulttime))+".jpg")
                    lock.release()
                    if qtag.get(True)=="reok":
                        self.answer=qdataresu.get(True)
                        result="时间："+str(time.strftime('%Y-%m-%d %H:%M:%S',resulttime))+"识别结果:"+self.answer
                        self.textEdit_5.append(result)
                        f.write(result+'\n')
            elif self.camera=="close":
                cap.release()
                self.label_8.setPixmap(QtGui.QPixmap())
                self.label_12.setText("摄像头已关闭")
                f.close()
                break
    def pushButtonopencameraReclicked(self):
        self.camera=""
        self.textEdit_5.clear()
        self.label_12.setText("摄像头正在开启...")
        th3 = threading.Thread(target=self.thread3, args=(self.qtag, self.qdata,self.qdataresu,self.lock))
        th3.start()
    def pushButtonclosecameraReclicked(self):
        self.camera="close"
    #######---------重写主窗口关闭函数
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '本程序',
                                               "是否要退出程序？",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.lock.acquire()
            self.qclose.put("close")
            self.lock.release()
            event.accept()
        else:
            event.ignore()
#####--------主窗体进程
def Ui(qtag,qdata,qdataresu,qclose,lock):
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form(qtag,qdata,qdataresu,qclose,lock)
    my_pyqt_form.show()
    sys.exit(app.exec_())
#####----------识别进程
def recognition(qtag, qdata,qdataresu,lock):
    while qtag.get(True) == "ok":
        file_path = qdata.get(True)
        output_file_path = qdata.get(True)
        print(file_path)
        print(output_file_path)
        im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        answer = face_recognition.picture_recognition(im, file_path, output_file_path)
        lock.acquire()  # 加上锁
        qdataresu.put(answer)
        qtag.put("reok")
        lock.release()  # 释放锁
#####-----------主进程
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    # 父进程创建Queue，并传给各个子进程：
    qtag = manager.Queue()
    qdata = manager.Queue()
    qdataresu = manager.Queue()
    qclose=manager.Queue()
    lock = manager.Lock()  # 初始化一把锁
    p = Pool()
    pw = p.apply_async(Ui, args=(qtag,qdata,qdataresu,qclose,lock))
    pr = p.apply_async(recognition, args=(qtag,qdata,qdataresu,lock))
    p.close()
    while qclose.get(True) == "close":
        sys.exit(0)
    p.join()
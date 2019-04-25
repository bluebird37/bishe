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
    #page2��������
    ##----����ͼƬʶ��
    file_path = ""
    output_file_path = ""
    #tag = 0#��־
    ##-----ѡ���ļ�������ʶ��
    PictureInfolder_path=""#ѡ��ͼƬ�����ļ���·��
    PictureOufolder_path=""#ѡ��ͼƬ����ļ���·��
    answer=""
    ##-----����ͷʵʱʶ��
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
        #####page1-----buttons�¼�
        self.pushButton.clicked.connect(self.pushButtonAddclicked)#ѡ�������ʶ���û��ļ����¼�
        self.pushButton_2.clicked.connect(self.pushButtonTrainclicked)#ѵ���¼�
        self.pushButton_3.clicked.connect(self.pushButtonPutclicked)#�ύ�¼�
        #####page2-----buttons�¼�----ͼƬʶ��
        self.pushButton_5.clicked.connect(self.pushButtonPictureFileclicked)  # ѡ��ͼƬ�¼�
        self.pushButton_6.clicked.connect(self.pushButtonPictureFileReclicked)  # ѡ��ͼƬʶ������¼�
        self.pushButton_7.clicked.connect(self.pushButtonPictureInfolderReclicked)  # ѡ��ͼƬ�����ļ���
        self.pushButton_8.clicked.connect(self.pushButtonPictureOufolderReclicked)  # ѡ��ͼƬ����ļ���
        self.pushButton_9.clicked.connect(self.pushButtonPicturefolderReclicked)  # ѡ��ͼƬ�ļ�������ʶ��
        #######page2-----buttons�¼�----����ͷʶ��
        self.pushButton_11.clicked.connect(self.pushButtonopencameraReclicked)  # ѡ��ͼƬ����ļ���
        self.pushButton_12.clicked.connect(self.pushButtonclosecameraReclicked)  # ѡ��ͼƬ�ļ�������ʶ��
    #ʵ��pushButton_click()������textEdit�����Ƿ���ȥ���ı����id
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
    ######page1-----buttons�¼�
    def pushButtonAddclicked(self):
        root = tkinter.Tk()  # ����һ��Tkinter.Tk()ʵ��
        root.withdraw()  # ��Tkinter.Tk()ʵ������
        default_dir = "C:/Users/27117/Desktop"
        folder_path = tkinter.filedialog.askdirectory(parent=root,
                                         title="ѡ���ļ���",
                                         initialdir=(os.path.expanduser(default_dir)))
        root.destroy()
        self.textEdit.setPlainText(folder_path)
    def pushButtonTrainclicked(self):
        print("train")
    def pushButtonPutclicked(self):
        print("put")
    #######page2-----buttons�¼�----ͼƬʶ��
    def pushButtonPictureFileclicked(self):
        root2 = tkinter.Tk()  # ����һ��Tkinter.Tk()ʵ��
        root2.withdraw()  # ��Tkinter.Tk()ʵ������
        default_dir = "C:/Users/27117/Desktop"
        my_filetypes = [('jpg files', '.jpg')]
        self.file_path = tkinter.filedialog.askopenfilename(title='ѡ���ļ�',
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
                    QMessageBox.warning(self, '����', '�ļ���ʽ����������ѡ���ļ�')
            else:
                QMessageBox.warning(self, '����', 'û��ѡ����jpg��β���ļ�')
        else:
            self.label_6.setPixmap(QtGui.QPixmap())

    def thread1(self, qtag, qdataresu):
        if qtag.get(True) == "reok":
            self.answer = qdataresu.get(True)
            if self.answer != "ͼ���������޷���⵽����":
                png2 = QtGui.QPixmap('./output_lingshi_picture/temp1.png')
                self.label_7.setPixmap(png2)
                self.label_7.setScaledContents(True)
                #self.tag = 0
                self.pushButton_6.setText("ʶ��")
            else:
                self.pushButton_6.setText("ʶ��")
                QMessageBox.warning(self, '����', 'ͼ������,��ѡ������ͼƬ')
    def thread2(self, qtag, qdata,qdataresu,lock):
        for (root, dirs, filenames) in os.walk(self.PictureInfolder_path):
            for filename in filenames:
                if filename[len(filename) - 3:] != "jpg":
                    self.textEdit_10.append("�ļ���" + filename + "����jpg�ļ�")
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
                            answer_textEdit_10_set = "�ļ���" + filename + "   ʶ����Ϊ��" + self.answer
                            self.textEdit_10.append(answer_textEdit_10_set)
                    else:
                        self.textEdit_10.append("�ļ���" + filename + "   ��ʽ����")
        self.textEdit_10.append("end!")
        self.textEdit_10.append("\n" + "����ɲ鿴�ļ��У�" + self.PictureOufolder_path + "/   ������ͼƬ")
        self.pushButton_9.setText("ʶ��")
    def pushButtonPictureFileReclicked(self):
        if self.pushButton_6.text()=="ʶ��":
            if len(self.textEdit_6.toPlainText())>0:
                self.pushButton_6.setText("ʶ����...")
                self.file_path=self.textEdit_6.toPlainText()
                self.output_file_path='./output_lingshi_picture/temp1.png'
                ####--------�������߳�
                self.lock.acquire()
                self.qtag.put("ok")
                self.qdata.put(self.file_path)
                self.qdata.put(self.output_file_path)
                self.lock.release()
                th =  threading.Thread(target=self.thread1,args=(self.qtag,self.qdataresu))
                th.start()
            else:
                QMessageBox.warning(self, '����', 'û��ѡ���ļ�')
    def pushButtonPictureInfolderReclicked(self):
        root3 = tkinter.Tk()  # ����һ��Tkinter.Tk()ʵ��
        root3.withdraw()  # ��Tkinter.Tk()ʵ������
        default_dir = "C:/Users/27117/Desktop"
        self.PictureInfolder_path = tkinter.filedialog.askdirectory(parent=root3,
                                                      title="ѡ���ļ���",
                                                      initialdir=(os.path.expanduser(default_dir)))
        root3.destroy()
        self.textEdit_8.setPlainText(self.PictureInfolder_path)
    def pushButtonPictureOufolderReclicked(self):
        root4 = tkinter.Tk()  # ����һ��Tkinter.Tk()ʵ��
        root4.withdraw()  # ��Tkinter.Tk()ʵ������
        default_dir = "C:/Users/27117/Desktop"
        self.PictureOufolder_path = tkinter.filedialog.askdirectory(parent=root4,
                                                               title="ѡ���ļ���",
                                                               initialdir=(os.path.expanduser(default_dir)))
        root4.destroy()
        self.textEdit_9.setPlainText(self.PictureOufolder_path)
    def pushButtonPicturefolderReclicked(self):
        if self.pushButton_9.text()=="ʶ��":
            self.textEdit_10.clear()
            self.textEdit_10.append("�ļ��У�"+self.PictureInfolder_path+"/��"+"\n")
            if len(self.textEdit_8.toPlainText())>0 and len(self.textEdit_9.toPlainText())>0:
                self.pushButton_9.setText("ʶ����...")
                th2 = threading.Thread(target=self.thread2, args=(self.qtag, self.qdata,self.qdataresu,self.lock))
                th2.start()
            elif len(self.textEdit_8.toPlainText())==0:
                QMessageBox.warning(self, '����', 'û��ѡ�������ļ���')
            elif len(self.textEdit_9.toPlainText())==0:
                QMessageBox.warning(self, '����', 'û��ѡ������ļ���')
    #######page2-----buttons�¼�----����ͷʶ��
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
                self.label_12.setText("����ͷ�ѿ���")
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
                        result="ʱ�䣺"+str(time.strftime('%Y-%m-%d %H:%M:%S',resulttime))+"ʶ����:"+self.answer
                        self.textEdit_5.append(result)
                        f.write(result+'\n')
            elif self.camera=="close":
                cap.release()
                self.label_8.setPixmap(QtGui.QPixmap())
                self.label_12.setText("����ͷ�ѹر�")
                f.close()
                break
    def pushButtonopencameraReclicked(self):
        self.camera=""
        self.textEdit_5.clear()
        self.label_12.setText("����ͷ���ڿ���...")
        th3 = threading.Thread(target=self.thread3, args=(self.qtag, self.qdata,self.qdataresu,self.lock))
        th3.start()
    def pushButtonclosecameraReclicked(self):
        self.camera="close"
    #######---------��д�����ڹرպ���
    def closeEvent(self, event):
        reply = QtWidgets.QMessageBox.question(self,
                                               '������',
                                               "�Ƿ�Ҫ�˳�����",
                                               QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.lock.acquire()
            self.qclose.put("close")
            self.lock.release()
            event.accept()
        else:
            event.ignore()
#####--------���������
def Ui(qtag,qdata,qdataresu,qclose,lock):
    app = QtWidgets.QApplication(sys.argv)
    my_pyqt_form = MyPyQT_Form(qtag,qdata,qdataresu,qclose,lock)
    my_pyqt_form.show()
    sys.exit(app.exec_())
#####----------ʶ�����
def recognition(qtag, qdata,qdataresu,lock):
    while qtag.get(True) == "ok":
        file_path = qdata.get(True)
        output_file_path = qdata.get(True)
        print(file_path)
        print(output_file_path)
        im = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        answer = face_recognition.picture_recognition(im, file_path, output_file_path)
        lock.acquire()  # ������
        qdataresu.put(answer)
        qtag.put("reok")
        lock.release()  # �ͷ���
#####-----------������
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    # �����̴���Queue�������������ӽ��̣�
    qtag = manager.Queue()
    qdata = manager.Queue()
    qdataresu = manager.Queue()
    qclose=manager.Queue()
    lock = manager.Lock()  # ��ʼ��һ����
    p = Pool()
    pw = p.apply_async(Ui, args=(qtag,qdata,qdataresu,qclose,lock))
    pr = p.apply_async(recognition, args=(qtag,qdata,qdataresu,lock))
    p.close()
    while qclose.get(True) == "close":
        sys.exit(0)
    p.join()
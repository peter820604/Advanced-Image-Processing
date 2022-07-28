#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os
if hasattr(sys, 'frozen'):
    os.environ['PATH'] = sys._MEIPASS + ";" + os.environ['PATH']
import cv2
import numpy
import cv2 as cv
from PyQt5 import QtCore, QtWidgets,QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (QApplication, QDialog,
                             QFileDialog, QGridLayout, QLabel, QPushButton, QInputDialog)
from pylab import *
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import sys
import math
import random

class win(QtWidgets.QDialog):
    def __init__(self,parent=None):

        # 初始二個img的ndarray用於存儲圖像
        self.img = np.ndarray(())
        self.img2 = np.ndarray(())
        self.img3 = np.ndarray(())
        self.img4 = np.ndarray(())
        self.ax = np.ndarray(())
        super(win, self).__init__(parent)
        self.initUI()

    def initUI(self):
        self.resize(500, 500)
        self.setWindowTitle('AIP 60847089s')
        self.btnOpen = QPushButton('選擇影像', self)
        self.btnSave = QPushButton('儲存影像', self)
        self.btnGray = QPushButton('灰階', self)
        self.btnGauss = QPushButton('灰階高斯白雜訊', self)
        self.btnHistogram = QPushButton('灰階直方圖', self)
        self.btnwaHaar = QPushButton('灰階Haar小波轉換', self)
        self.btnequalization = QPushButton('直方圖均化', self)
        self.btnsmoothing = QPushButton('平滑化', self)
        self.btnedgedetectors = QPushButton('邊緣偵測', self)
        self.label = QLabel()
        self.label.setFixedWidth(350)
        self.label.setFixedHeight(350)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)  # 放figure圖
        self.label2 = QLabel()
        self.label2.setFixedWidth(350)
        self.label2.setFixedHeight(350)
        self.figure2 = plt.figure()
        self.canvas2 = FigureCanvas(self.figure2)  # 放figure圖

        # 佈局
        layout = QGridLayout(self)
        layout.addWidget(self.label, 1, 1, 4, 4)
        layout.addWidget(self.canvas, 1, 4, 4, 6)
        layout.addWidget(self.label2, 5, 1, 4, 4)
        layout.addWidget(self.canvas2, 5, 4, 4, 6)
        layout.addWidget(self.btnOpen, 0, 1)
        layout.addWidget(self.btnSave, 0, 2)
        layout.addWidget(self.btnGray, 0, 3)
        layout.addWidget(self.btnHistogram, 0, 4)
        layout.addWidget(self.btnGauss, 0, 5)
        layout.addWidget(self.btnwaHaar, 0, 6)
        layout.addWidget(self.btnequalization, 0, 7)
        layout.addWidget(self.btnsmoothing, 0, 8)
        layout.addWidget(self.btnedgedetectors, 0, 9)

        # 連接
        self.btnOpen.clicked.connect(self.openSlot)
        self.btnSave.clicked.connect(self.saveSlot)
        self.btnGray.clicked.connect(self.gray)
        self.btnGauss.clicked.connect(self.Gauss)
        self.btnHistogram.clicked.connect(self.histogram)
        self.btnwaHaar.clicked.connect(self.Haar)
        self.btnequalization.clicked.connect(self.equalization)
        self.btnequalization.clicked.connect(self.equalization)
        self.btnsmoothing.clicked.connect(self.smoothing)
        self.btnedgedetectors.clicked.connect(self.edgedetectors)

    def openSlot(self):

        self.img = np.ndarray(())
        self.img2 = np.ndarray(())
        self.img3 = np.ndarray(())
        self.img4 = np.ndarray(())

        fileName, tmp = QFileDialog.getOpenFileName(self,"打開影像","","*.jpg;;*.bmp;;*.ppm;;All Files (*)")
        if fileName is '':
            return
        self.img = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.img = cv2.resize(self.img, (512, 512), interpolation=cv2.INTER_CUBIC)

        self.img2 = self.img.copy()

        self.refreshShow()

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

    def saveSlot(self):

        if self.img2.size == 1:
            return




        list = ["上圖", "下圖"]

        style, okPressed = QInputDialog.getItem(self, "選擇", "", list)


        if style == "上圖":
            n = self.img2
        else:
            n = self.img4

        # 調用存儲文件dialog
        fileName, tmp = QFileDialog.getSaveFileName(self, '儲存影像', '', "*.jpg;;*.bmp;;*.ppm;;",)

        if fileName is '':
            return

        cv.imwrite(fileName, n)

    def gray(self):
        if self.img.size == 1:
            return

        self.img2 = self.img.copy()

        #判斷是否維灰階圖
        width = self.img.shape[0]
        height = self.img.shape[1]
        for x in range(0, int(width)):
            for y in range(0, int(height)):
                r, g, b = self.img[x, y]
                r = int(r)
                g = int(g)
                b = int(b)
                if (r != g) and (g != b):
                    return self.gray2()
                else:
                    pass

        self.refreshShow()

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

    def gray2(self):

        b, g, r = cv2.split(self.img2)
        bb = b // 3 + g // 3 + r // 3
        gg = b // 3 + g // 3 + r // 3
        rr = b // 3 + g // 3 + r // 3

        self.img2 = cv2.merge([bb, gg, rr])
        self.refreshShow()

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

    def histogram(self):

        if self.img2.size == 1:
            return

        height = self.img2.shape[0]
        width = self.img2.shape[1]
        c_a = []
        #掃每個pixel之灰階值
        for x in range(height):
            for y in range(width):
                c = self.img2[int(x), int(y)]
                c_a.append(c[0])
        n = []
        for x in range(256):
            c = x
            n.append(c)
        self.ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8]) #canvas大小
        self.ax.clear()  #每次跑圖前清空
        self.ax.hist(c_a, bins=n)        #ax.hist(data,x軸)
        plt.title('Gray Scale Histogram')


        self.canvas.draw()

    def Gauss(self):
        if self.img.size == 1:
            return
        #標準差輸入
        std, okPressed = QInputDialog.getInt(self, "輸入","輸入標準差(1~)", 0, min = 1,step = 1)

        #如直接關掉視窗直接判定輸入為0
        if std == 0:
            return

        self.img2 = self.img.copy()
        self.gray()  # 轉灰階

        img = self.img2

        height = img.shape[0]
        width = img.shape[1]
        depth = img.shape[2]

        gray_img = numpy.zeros([height, width, depth], numpy.uint8)

        for column in range(0, width, 2):
            for row in range(height):
                for chan in range(depth):
                    r = random.random()
                    v = random.random()
                    std = std
                    z1 = std * math.cos(2 * math.pi * v) * math.sqrt(-2 * math.log(r, 10))
                    z2 = std * math.sin(2 * math.pi * v) * math.sqrt(-2 * math.log(r, 10))

                    new_valuer = img[int(row), int(column)][1]
                    new_valuer2 = img[int(row), int(column + 1)][1]

                    fx = new_valuer + z1
                    hx = new_valuer2 + z2

                if fx < 0:
                    new_valuer = 0
                elif fx > 255:
                    new_valuer = 255
                else:
                    new_valuer = fx

                if hx < 0:
                    new_valuer2 = 0
                elif hx > 255:
                    new_valuer2 = 255
                else:
                    new_valuer2 = hx

                gray_img[row, column, :] = new_valuer
                gray_img[row, column + 1, :] = new_valuer2

        self.img2 = gray_img
        self.refreshShow()

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

    def Haar(self):
        if self.img.size == 1:
            return

        #層數輸入
        n, okPressed = QInputDialog.getInt(self, "輸入","輸入轉換層數(1~9)", 0, 0, 9, 1)

        self.gray() #轉灰階

        self.img3 = self.img2.copy()

        R = self.img3
        n = n
        for i in range(0, n):
            a1, a2, a3 = R.shape

            a1 = a1 // (2 ** i)
            a2 = a2 // (2 ** i)
            # LL
            a = (R[:a1, 0:a2:2]) // 2 + (R[:a1, 1:a2:2]) // 2
            a4, a5, a6 = a.shape
            aa = (a[0:a4:2, :a5]) // 2 + (a[1:a4:2, :a5]) // 2

            # LH
            b = (R[:a1, 0:a2:2]) // 2 - (R[:a1, 1:a2:2]) // 2
            b4, b5, b6 = b.shape
            bb = (b[0:b4:2, :b5]) // 2 + (b[1:b4:2, :b5]) // 2
            # HL
            c = (R[:a1, 0:a2:2]) // 2 + (R[:a1, 1:a2:2]) // 2
            c4, c5, c6 = c.shape
            cc = (c[0:c4:2, :c5]) // 2 - (c[1:c4:2, :c5]) // 2
            # HH
            d = (R[:a1, 0:a2:2]) // 2 - (R[:a1, 1:a2:2]) // 2
            d4, d5, d6 = d.shape
            dd = (d[0:d4:2, :d5]) // 2 - (d[1:d4:2, :d5]) // 2

            cv2.rectangle(aa, (0, 0), (a1, a2), (255, 255, 255), 1)
            cv2.rectangle(bb, (0, 0), (a1, a2), (255, 255, 255), 1)
            cv2.rectangle(cc, (0, 0), (a1, a2), (255, 255, 255), 1)
            cv2.rectangle(dd, (0, 0), (a1, a2), (255, 255, 255), 1)


            A = np.hstack([aa, bb])
            B = np.hstack([cc, dd])
            C = np.vstack((A, B))
            R[:a1, :a2] = C

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

        self.refreshShow3()

    def equalization(self):
        if self.img.size == 1:
            return

        self.gray() #轉灰階
        self.img3 = self.img2.copy()
        a, b, c = self.img2.shape
        result = []
        res = [0 for i in range(256)]
        res1 = [0 for i in range(256)]
        res2 = [0 for i in range(256)]
        res3 = [0 for i in range(256)]
        c_a = []
        c_a2 = []

        n = []
        for x in range(256):
            c = x
            n.append(c)

        # 灰階值統計
        for x in range(a):
            for y in range(b):
                c = self.img2[int(x), int(y)]
                c_a.append(c[0])
        for i in c_a:
            res[i] += 1
        # 值均化
        for i in range(0, 256, 1):
            res2[i] = res2[i - 1] + res[i]

        self.ax = self.figure.add_axes([0.1, 0.1, 0.8, 0.8]) #canvas大小
        self.ax.clear()  #每次跑圖前清空
        self.ax.hist(c_a, bins=n)        #ax.hist(data,x軸)
        self.canvas.draw()

        # 找第一個數量不為零的像素值

        for i in range(0, 256, 1):
            res2[i] != 0
            hcmin = i
            break

        for i in range(0, 256, 1):
            res3[i] = round((res2[i] - hcmin) / (a * b - hcmin) * 255)

        for x in range(a):
            for y in range(b):
                self.img3[x, y] = res3[self.img2[x, y][0]]

        # 灰階值統計
        for x in range(a):
            for y in range(b):
                c = self.img3[int(x), int(y)]
                c_a2.append(c[0])
        for i in c_a2:
            res1[i] += 1

        self.ax2 = self.figure2.add_axes([0.1, 0.1, 0.8, 0.8]) #canvas大小
        self.ax2.hist(c_a2, bins=n)        #ax.hist(data,x軸)
        self.canvas2.draw()

        self.refreshShow3()

    def smoothing(self):
        if self.img.size == 1:
            return

        # mask大小
        n, okPressed = QInputDialog.getInt(self, "輸入", "mask size", 0, min=3, step=2)

        #如直接關掉視窗直接判定輸入為0
        if n == 0:
            return


        self.gray()  # 轉灰階

        self.img3 = self.img2.copy()
        img = self.img3

        n = n
        m = int(n / 2)
        img2 = cv2.copyMakeBorder(img, m, m, m, m,
                                  cv.BORDER_REPLICATE)
        a, b, c = img2.shape
        aa, bb, cc = img.shape
        img3 = numpy.zeros([aa, bb, 3], numpy.int)
        for i in range(0, a - n + 1, 1):
            for j in range(0, a - n + 1, 1):
                for x in range(0, n, 1):
                    for y in range(0, n, 1):
                        fun = i + x
                        funn = j + y
                        h = (img2[fun, funn]).astype(np.int)
                        img3[i, j, :] += h[0]
        img[:, :, :] = img3[:, :, :] / (n ** 2)

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

        self.refreshShow3()

    def edgedetectors(self):
        if self.img.size == 1:
            return

        if self.img3.size == 1:
           self.img3 = self.img.copy()
        else:
            pass

        # mask大小
        n, okPressed = QInputDialog.getInt(self, "輸入", "mask size", 0, min=3, step=2)

        #如直接關掉視窗直接判定輸入為0
        if n == 0:
            return

        tr, okPressed = QInputDialog.getInt(self, "輸入", "Threshold", 0, min=0, step=1)

        self.img4 = self.img3.copy()
        img = self.img4

        n = n
        m = int(n / 2)
        tr = tr
        img2 = cv2.copyMakeBorder(img, m, m, m, m,
                                  cv.BORDER_REPLICATE)
        a, b, c = img2.shape
        aa, bb, cc = img.shape
        img3 = numpy.zeros([aa, bb, 3], numpy.int)

        for i in range(0 + m, a - m, 1):
            for j in range(0 + m, a - m, 1):
                for x in range(-m, m, 1):
                    row = i + x
                    clum = j + x
                    # 垂直
                    ve = (img2[i - 1, clum]).astype(np.int)
                    ve2 = (img2[i + 1, clum]).astype(np.int)
                    ve3 = abs(ve - ve2)

                    # 水平
                    hy = (img2[row, j - 1]).astype(np.int)
                    hy2 = (img2[row, j + 1]).astype(np.int)
                    hy3 = abs(hy - hy2)

                    img3[i - m, j - m, :] += hy3[0]
                    img3[i - m, j - m, :] += ve3[0]

                if img3[i - m, j - m][0] > tr:
                    img3[i - m, j - m, :] = 150
                else:
                    img3[i - m, j - m, :] = 250


        self.img4[:, :, :] = img3[:, :, :]

        # 每次跑圖前清空
        self.label2.clear()
        self.figure.clear()
        self.canvas.draw()
        self.figure2.clear()
        self.canvas2.draw()

        self.refreshShow4()

    def refreshShow(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img2.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(self.img2.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        # 將Qimage調整為適合label大小
        self.label.setScaledContents(True)

    def refreshShow2(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img2.shape
        bytesPerLine = 3 * width
        self.qImg2 = QImage(self.img2.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label2.setPixmap(QPixmap.fromImage(self.qImg2))
        # 將Qimage調整為適合label大小
        self.label2.setScaledContents(True)


    def refreshShow3(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img3.shape
        bytesPerLine = 3 * width
        self.qImg3 = QImage(self.img3.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label2.setPixmap(QPixmap.fromImage(self.qImg3))
        # 將Qimage調整為適合label大小
        self.label2.setScaledContents(True)

    def refreshShow4(self):
        #提取圖像的尺寸和通道, 用於將opencv下的image轉換成Qimage
        height, width, channel = self.img4.shape
        bytesPerLine = 3 * width
        self.qImg4 = QImage(self.img4.data, width, height, bytesPerLine,QImage.Format_RGB888).rgbSwapped()

        # 將Qimage顯示出來
        self.label2.setPixmap(QPixmap.fromImage(self.qImg4))
        # 將Qimage調整為適合label大小
        self.label2.setScaledContents(True)


if __name__ == '__main__':
    a = QApplication(sys.argv)
    w = win()
    w.show()
    sys.exit(a.exec_())
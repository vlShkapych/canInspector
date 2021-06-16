import cv2 as cv
from PyQt5 import QtGui, QtCore, QtWidgets


class ComputerVision(object):
    def __init__(self):
        self.cap = cv.VideoCapture(0);
    

    def getFrameQt(self):
        _,img = self.cap.read()
        img = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        return img;
# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
import RPi.GPIO as GPIO
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMainWindow
import cv2
from PyQt5 import QtGui,QtCore
from PyQt5.QtCore import Qt
import pandas as pd

from ui.Ui_mainWindow import Ui_MainWindow

import inspector 

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
       
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        
        def inspectionSubscription(frame,updatedInfo):
            self.camera.setPixmap(QtGui.QPixmap(frame));
            self.textBox_canInspected.setText(updatedInfo[0]);
            self.textBox_canRejected.setText(updatedInfo[1]);

            model = TableModel(updatedInfo[2])
            self.tableJournal.setModel(model)
            
        def stateInspection(state):
            self.textBox_status.setText(state[0]);
            self.textBox_machinePerfomance.setText(state[1]);




        self.inspector = inspector.Inspector();
        self.emergencyStop.mousePressEvent = self.inspector.emergencyStop_push;

        self.inspector.subscription = inspectionSubscription;
        self.inspector.updateInterface = stateInspection;
        
    
    




    @pyqtSlot()
    def on_onOffButton_clicked(self):


        GPIO.setup(self.inspector._canSensor_channel, GPIO.IN, pull_up_down=GPIO.PUD_OFF)
        

        cap = cv2.VideoCapture(0);

        cv2.namedWindow('settings')
        cv2.resizeWindow("settings", 640, 420)  

        cv2.createTrackbar('PaddingTop','settings',0,420,lambda position: self.inspector.setFrame(0,position)) # Hue is from 0-179 for Opencv
        cv2.setTrackbarPos('PaddingRight', 'settings', 0)
        cv2.createTrackbar('PaddingRight','settings',0,420,lambda position: self.inspector.setFrame(1,position)) # Hue is from 0-179 for Opencv
        cv2.setTrackbarPos('PaddingRight', 'settings', 0)
        cv2.createTrackbar('PaddingBottom','settings',0,420,lambda position: self.inspector.setFrame(2,position)) # Hue is from 0-179 for Opencv
        cv2.setTrackbarPos('PaddingBottom', 'settings', 0)
        cv2.createTrackbar('PaddingLeft','settings',0,420,lambda position: self.inspector.setFrame(3,position)) # Hue is from 0-179 for Opencv
        cv2.setTrackbarPos('PaddingLeft', 'settings', 0)
        cv2.createTrackbar('Mode(AI/LASER)','settings',0,1,lambda position: self.inspector.setMode(position)) # Hue is from 0-179 for Opencv
        cv2.setTrackbarPos('Mode(AI/LASER)', 'settings', 0)
        while(1):
            ret,img = cap.read();
            img = cv2.resize(img, (300, 300))

            cv2.imshow('settings',self.inspector._framePreprocess(img));
            
            if cv2.waitKey(33)& 0xFF == ord('q'):
                break
        self.textBox_mode.setText(self.inspector._settings[0]);

        cv2.destroyAllWindows()
        GPIO.setup(self.inspector._canSensor_channel, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
    
    @pyqtSlot()
    def on_kvitButton_clicked(self):
        data = pd.DataFrame([
          [1, 9, 2],
          [1, 0, -1],
          [3, 5, 2],
          [3, 3, 2],
          [5, 8, 9],
        ], columns = ['A', 'B', 'C'], index=['Row 1', 'Row 2', 'Row 3', 'Row 4', 'Row 5'])

        self.model = TableModel(data)
        self.tableJournal.setModel(self.model)
    
    
class TableModel(QtCore.QAbstractTableModel):

    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            value = self._data.iloc[index.row(), index.column()]
            return str(value)

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Vertical:
                return str(self._data.index[section])




        

       

        
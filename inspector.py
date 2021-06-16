from numpy.lib.function_base import select
import RPi.GPIO as GPIO
import time 
import cv2
from PyQt5 import QtGui, QtCore, QtWidgets
from AI_Inspector import AI_Inspector
from Laser_Inspector import Laser_Inspector
import pandas as pd
from datetime import datetime

class Inspector(object):

    def __init__(self):
        self._state = ["Stand By","0"];
        self._inspected = "00000000000";
        self._defective = "00000000000";
        
        self.aiInspector = AI_Inspector();
        self.laserInspector = Laser_Inspector();
        
        self.inspector = self.laserInspector;

        self._settings = ["AI",[0,0,0,0]];

        self._journal = []

        self.GPIO_init();

    

    def setFrame(self,index,value):
        self._settings[1][index] = value;
    
    def setMode(self,position):
        if position == 0:
            self.inspector = self.aiInspector;
            self._settings[0] = "AI";
        else:
            self.inspector = self.laserInspector;
            self._settings[0] = "Laser";

    def _framePreprocess(self,frame):
            
        # frame[0:self._settings[1][0],:] = 0
        # frame[len(frame)-self._settings[1][2]:,:] = 0
        # frame[:,0:self._settings[1][3]] = 0
        # frame[:,len(frame[0])-self._settings[1][1]:] = 0        
    
        return frame[self._settings[1][0]:len(frame)-self._settings[1][2],self._settings[1][3]:len(frame[0])-self._settings[1][1]];

    def signalGenerator(self,pin,delay):
        
        GPIO.output(pin,GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(pin,GPIO.LOW)

        
    def inspectedCounter(self,isDefective):
        inspected = int(self._inspected);
        defective = int(self._defective);
        
        inspected += 1;

        if(isDefective == 1):
            defective+=1;

        defective = str(defective);
        inspected = str(inspected);

        appendix = "".join(["0" for i  in range(len(defective),11)]);
        defective = appendix + defective;

        appendix = "".join(["0" for i  in range(len(inspected),11)]);
        inspected = appendix + inspected;        

        self._inspected = inspected;
        self._defective = defective;
        
    

    def GPIO_init(self):
        self._canOk_channel = 21;
        self._canDefective_channel = 20;
        
        self._kvit_channel = 16;

        self._emergencyStop_channel1 = 26;
        self._emergencyStop_channel2 = 19;

        self._machineReady_channel = 13;

        self._canSensor_channel = 4;

        self._machineTact = 16;
        self._laserPin = 26;
        self._flashPin = 19;


        self.subscription = None;


        GPIO.setmode(GPIO.BCM);
        GPIO.setwarnings(False);


        GPIO.setup(self._laserPin, GPIO.OUT);
        GPIO.setup(self._flashPin, GPIO.OUT);

        GPIO.setup(self._canOk_channel, GPIO.OUT);
        GPIO.setup(self._machineTact, GPIO.OUT);
        GPIO.setup(self._canDefective_channel, GPIO.OUT);
        GPIO.setup(self._kvit_channel, GPIO.OUT);
        GPIO.setup(self._emergencyStop_channel1, GPIO.OUT);
        GPIO.setup(self._emergencyStop_channel2, GPIO.OUT);
        
        GPIO.setup(self._machineReady_channel, GPIO.OUT);


        GPIO.setup(self._canSensor_channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        GPIO.add_event_detect(self._canSensor_channel, GPIO.RISING, 
            callback=self.canDetected, bouncetime=2000)
    



    def canDetected(self,_):

        GPIO.output(self._machineTact,GPIO.HIGH);
        if self._settings[0] == "Laser":
            GPIO.output(self._laserPin,GPIO.HIGH);
        else:
            GPIO.output(self._flashPin,GPIO.HIGH);
        cap = cv2.VideoCapture(0);
        _,frame = cap.read();
        frame = cv2.resize(frame, (300, 300))
        nframe = self._framePreprocess(frame)
        start = time.time()

        


        answer = self.inspector.inspect(nframe);
        if self._settings[0] == "Laser":
            GPIO.output(self._laserPin,GPIO.LOW);
        else:
            GPIO.output(self._flashPin,GPIO.LOW);

        end = time.time()

        cph = round(1/(end - start))*3600;
 
        image = QtGui.QImage(answer[0].data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()

      
        self.inspectedCounter(answer[1]);
        
        r = lambda: "Defected" if (answer[1]==1) else "Passed"

        self._journal.append([int(self._inspected),datetime.now().strftime("%H:%M:%S"),r()]);

        table = pd.DataFrame(self._journal, columns = ['#', 'Time', 'Check Result'], index=[i for i in range(int(self._inspected))])
        

        self._state[1] = "{}c/h".format(cph)
        self.updateInterface(self._state)

        self.subscription(image,[self._inspected, self._defective,table,]);
        if answer[1] == 0:
            self.canOk();
        else:
            self.canDefective();
        GPIO.output(self._machineTact,GPIO.LOW);

     
        
    def kvit_push(self):
        if(self._state == "fault"):
            self._state == "normal";
        
        GPIO.output(self._emergencyStop_channel1,GPIO.HIGH);
        GPIO.output(self._emergencyStop_channel1,GPIO.HIGH);

        self.signalGenerator(self._kvit_channel,0.8);
        

    def emergencyStop_push(self,event):
        GPIO.output(self._emergencyStop_channel1,GPIO.LOW);
        GPIO.output(self._emergencyStop_channel1,GPIO.LOW);
        self._state[0] = "Fault";
        self.updateInterface(self._state)

    def machineState(self,state):
        if(state == 1):
            GPIO.output(self._machineReady_channel,GPIO.HIGH);
        elif(state == 0):
            GPIO.output(self._machineReady_channel,GPIO.LOW);

    



    def canOk(self):
         self.signalGenerator(self._canOk_channel,0.8);
        
    def canDefective(self):
        self.signalGenerator(self._canDefective_channel,0.8);


    

    

    


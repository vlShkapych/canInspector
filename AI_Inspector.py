import os
import argparse
import cv2
import numpy as np
import sys
import glob
import importlib.util
from tensorflow.lite.python.interpreter import Interpreter

class AI_Inspector(object):

    def __init__(self):
        self._aiModel = "/home/pi/canInspector/program/my_ssd_mobnet/detect.tflite";
        self._aiLabels = "/home/pi/canInspector/program/my_ssd_mobnet/labelmap.txt";
        self._minConfThreshold = 0.5;
        with open(self._aiLabels , 'r') as f:
            self._labels = [line.strip() for line in f.readlines()];
        self._interpreter = Interpreter(model_path=self._aiModel);
        self._interpreter.allocate_tensors();

        self._input_details = self._interpreter.get_input_details();
        self._output_details = self._interpreter.get_output_details();
        self._height = self._input_details[0]['shape'][1];
        self._width = self._input_details[0]['shape'][2];
        self._floating_model = (self._input_details[0]['dtype'] == np.float32);
        self._input_mean = 127.5;
        self._input_std = 127.5;

    def _prepareFrame(self,frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB);
        imH, imW, _ = frame.shape;
        image_resized = cv2.resize(image_rgb, (self._width, self._height));
        input_data = np.expand_dims(image_resized, axis=0);
        if(self._floating_model):
            input_data = (np.float32(input_data) - self._input_mean) / self._input_std;
        return input_data;
       
    def _aiResolution(self,ai,frame):
        boxes = ai[0]
        classes = ai[1] 
        scores = ai[2]
        
        image = frame;
        imH, imW, _ = image.shape;

        defected = False;

        for i in range(len(scores)):
            if ((scores[i] > self._minConfThreshold) and (scores[i] <= 1.0)):
                defected = True;

                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                    
                cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = self._labels[int(classes[i])] # Look up object name from "labels" array using class index
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
        return [image,defected];


    def inspect(self,frame):
        frameAI = self._prepareFrame(frame);
        self._interpreter.set_tensor(self._input_details[0]['index'], frameAI);
        self._interpreter.invoke();

        boxes = self._interpreter.get_tensor(self._output_details[0]['index'])[0];
        classes = self._interpreter.get_tensor(self._output_details[1]['index'])[0]; 
        scores = self._interpreter.get_tensor(self._output_details[2]['index'])[0];

        return self._aiResolution([boxes,classes,scores],frame);

import numpy as np
from pandas import read_csv
from scipy.optimize import curve_fit
import cv2
import time

class Laser_Inspector(object):
    
    def __init__(self):
        self.threshold = 0.1;






    def _derivative(self,row):
        dx = [0]
        y = np.array(row,dtype=int)
        for i in range(1,len(y)-1):
            n = y[i+1] - y[i-1]
            dx.append(n)
        dx.append(0)
        return dx


    def inspect(self, frame):
        


        exp = [];
        x = [i for i in range(len(frame[0]))];
        frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        def objective(x, a, b, c, d, e, f):
            return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f
        start = time.time()
        for num,row in enumerate(frameGray):
        
            y = row
            
            y = self._derivative(row) #
            y = np.array(y)

            yMax = np.where(y == y.max())[0][0]
            yMin = np.where(y == y.min())[0][0]
            
            px = (yMax+yMin) // 2

            frame[num,:] = [0,0,0]
            frame[num,px] = [0,0,255]


            exp.append(px)
        x = [i for i in range(len(exp))]
        y = exp
        popt, _ = curve_fit(objective, x, y)
        a, b, c, d, e, f = popt
        end = time.time()
        sum = abs(a)+abs(b)+abs(c)+abs(d)+abs(e);
        print("(",sum,"):",a,b,c,d,e,f)

        return[frame,sum > self.threshold]






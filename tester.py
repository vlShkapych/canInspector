import cv2;
import time;
from pandas import read_csv
from scipy.optimize import curve_fit
import numpy as np;

cap = cv2.VideoCapture(0);

cv2.namedWindow('LaserDetector')



w=420
h=920
# video recorder
fourcc = cv2.VideoWriter_fourcc(*"XVID")
video_writer = cv2.VideoWriter("output.avi", fourcc, 2, (w, h))


def derivative(row):
        dx = [0]
        y = np.array(row,dtype=int)
        for i in range(1,len(y)-1):
            n = y[i+1] - y[i-1]
            dx.append(n)
        dx.append(0)
        return dx
def objective(x, a, b, c, d, e, f):
            return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f;

while True:
    _, frame = cap.read();
    frame = cv2.resize(frame, (420, 920));
    exp = [];
    x = [i for i in range(len(frame[0]))];
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);

    start = time.time()
    for num,row in enumerate(frameGray):
        
        y = row
            
        #     y = derivative(row) #
        #     y = np.array(y)

        #     yMax = np.where(y == y.max())[0][0]
        #     yMin = np.where(y == y.min())[0][0]
            
        #     px = (yMax+yMin) // 2
            
            #frame[num,:] = [0,0,0]
        px = np.where(y == y.max())[0][0]
        frame[num,px-4:px+4] = [100,200,100]
        exp.append(px)

    x = [i for i in range(len(exp))]
    y = exp
    popt, _ = curve_fit(objective, x, y)
    a, b, c, d, e, f = popt
    sum = abs(a)+abs(b)+abs(c)+abs(d)+abs(e);
    
    end = time.time()

    fps = 1/(end - start)
    fps = "FPS:"+str(round(fps,2));
    sum = "SUM:"+str(round(sum,3));

    font = cv2.FONT_HERSHEY_SIMPLEX


    cv2.putText(frame,sum,(20,50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame,fps,(20,100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    video_writer.write(frame)
    cv2.imshow('LaserDetector',frame)
    
    if cv2.waitKey(33)& 0xFF == ord('q'):
                break

cap.release()
video_writer.release()
cv2.destroyAllWindows()

import pandas as pd
import cv2 as cv
import torch
import numpy as np

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
            (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)


# model = torch.hub.load('ultralytics/yolov5', 'custom', 'runs/train/exp2/weights/best.pt')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

dt = 1.0/60
F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
H = np.array([1, 0, 0]).reshape(1, 3)
Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
R = np.array([0.5]).reshape(1, 1)
kf_x = KalmanFilter(F=F, H=H, Q=Q, R=R)
kf_y = KalmanFilter(F=F, H=H, Q=Q, R=R)

locx_base = []
locy_base = []

vidcap = cv.VideoCapture('../video/walk.mp4')
success, image = vidcap.read()
count = 0
predictions_x = []
predictions_y = []
while success:
    results = model(image)
    # results.print()
    # print(results.pandas().xyxy[0])
    
    locations = results.pandas().xyxy[0]
    try:
        locations = results.pandas().xyxy[0]
        obj = locations.loc[locations['name'] == 'person']
        obj_x = obj.iloc[0]['xmin']
        obj_y = obj.iloc[0]['ymin']
        obj_x = float(obj_x)
        obj_y = float(obj_y)
        # print('flame=%d'%count,end=' ')
        # print(obj_x, obj_y)
        locx_base.append(obj_x)
        locy_base.append(obj_y)
        x_arr = np.array(locx_base)
        y_arr = np.array(locy_base)
    except:
        print('Obj doesn\'t exist in this flame.')

    predictions_x.append(float(np.dot(H, kf_x.predict())[0]))
    predictions_y.append(float(np.dot(H, kf_y.predict())[0]))
    kf_x.update(obj_x)
    kf_y.update(obj_y)

    print('flame=%d'%count,end=' ')
    print(obj_x, obj_y, float(np.dot(H, kf_x.predict())[0]), float(np.dot(H, kf_y.predict())[0]))       


    
    success, image = vidcap.read()
    count += 1


import matplotlib.pyplot as plt
plt.plot(x_arr, label='Measurements')
plt.plot(predictions_x, label='Kalman Filter Prediction')
plt.xlabel('flames')
plt.ylabel('pixels')
plt.title('X-index Kalman Predictions')
plt.legend()
plt.savefig('../assets/index_x.png',dpi=1000)
plt.close()

plt.plot(y_arr, label='Measurements')
plt.plot(predictions_y, label='Kalman Filter Prediction')
plt.xlabel('flames')
plt.ylabel('pixels')
plt.title('Y-index Kalman Predictions')
plt.legend()
plt.savefig('../assets/index_y.png',dpi=1000)
plt.close()

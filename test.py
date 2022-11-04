import cv2 as cv
import numpy as np

path = '../assets/Kalman.mp4'
vidcap = cv.VideoCapture(path)
success, image = vidcap.read()
print(success)

# cv.imshow('../assets/index_x.png')



import numpy as np
import cv2

frame = cv2.imread("057_scene1_nm_H_105_2-rgb/51.png")
previous_frame = cv2.imread("057_scene1_nm_H_105_2-rgb/50.png")

hsv = np.zeros_like(previous_frame)
hsv[...,1] = 255

prvs = cv2.cvtColor(previous_frame,cv2.COLOR_BGR2GRAY)
next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
# print(flow.shape)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
hsv[...,0] = ang*180/np.pi/2
hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
rgb = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

# of = of * 100
# of1 = numpy.abs(of[1])
# of0 = numpy.abs(of[0])
cv2.imwrite("4.png",rgb)
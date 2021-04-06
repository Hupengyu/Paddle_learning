import cv2
import numpy as np
from matplotlib import pyplot as plt
import math

pdf_path = './pictures/ellispe.png'
img = cv2.imread(pdf_path)
# img=cv2.blur(img,(1,1))
imgray = cv2.Canny(img, 600, 100, 3)  # Canny边缘检测，参数可更改
# cv2.imshow("0",imgray)
ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
image, contours = cv2.findContours(thresh, cv2.RETR_TREE,
                                              cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
for cnt in contours:
    if len(cnt) > 50:
        S1 = cv2.contourArea(cnt)
        ell = cv2.fitEllipse(cnt)
        S2 = math.pi * ell[1][0] * ell[1][1]
        if (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。
            img = cv2.ellipse(img, ell, (0, 255, 0), 2)
            print(str(S1) + "    " + str(S2) + "   " + str(ell[0][0]) + "   " + str(ell[0][1]))
cv2.imshow("0", img)
cv2.waitKey(0)

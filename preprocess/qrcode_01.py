import cv2
import os
import numpy as np

pwd = os.getcwd()

lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])

img_path = os.path.join(pwd + "/pictures/" + "Image_00001.jpg")
res_path = os.path.join(pwd + "/results/" + "qrcode.jpg")

input_img = cv2.imread(img_path)
hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv', hsv)
cv2.waitKey()

mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('mask', mask)
# 4.只保留原图中的蓝色部分
res = cv2.bitwise_and(input_img, input_img, mask=mask)
cv2.imshow('blue', res)
cv2.waitKey()

gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', gray)
cv2.waitKey()

gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# cv2.imshow("gradient",gradient)
# 原本boundingRect没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
(_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
# cv2.imshow("thresh",thresh)
#
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("closed",closed)
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)

# img, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
# rect = cv2.minAreaRect(c)

for box in c:
    rect = cv2.minAreaRect(box)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(input_img, [box], -1, (0, 255, 0), 1)


cv2.imwrite(res_path, input_img)


if __name__ == '__main__':
    # 读取图片
    img_path = 'pictures/1.png'
    img = cv2.imread(img_path)
    
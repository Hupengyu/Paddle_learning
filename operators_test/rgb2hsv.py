import cv2
import numpy as np


def nothing(x):
    pass


# use track bar to perfectly define (1/2)
# the lower and upper values for HSV color space(2/2)
cv2.namedWindow("Tracking")
# 参数：1 Lower/Upper HSV 3 startValue 4 endValue
cv2.createTrackbar("LH", "Tracking", 35, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 43, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 46, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 77, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

while True:
    image_path = 'pictures/Image_00083.jpg'
    frame = cv2.imread(image_path)
    # frame = np.asarray(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_g = np.array([l_h, l_s, l_v])  # lower green value
    u_g = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, l_g, u_g)

    res = cv2.bitwise_and(frame, frame, mask=mask)  # src1,src2

    frame = cv2.resize(frame, None, fx=1, fy=1)
    cv2.imshow("frame", frame)
    mask = cv2.resize(mask, None, fx=1, fy=1)
    cv2.imshow("mask", mask)
    res = cv2.resize(res, None, fx=1, fy=1)
    cv2.imshow("res", res)
    key = cv2.waitKey(1)
    if key == 27:  # Esc
        break

cv2.destroyAllWindows()

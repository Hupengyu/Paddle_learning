import cv2
import numpy as np


def seal_mask_handle(img):
    img = cv2.resize(img, None, fx=0.5, fy=0.5,
                     interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 颜色空间转换

    low_range = np.array([0, 43, 46])  # 设定红色的低阈值
    high_range = np.array([10, 255, 255])  # 设定红色的高阈值

    low_range1 = np.array([156, 43, 46])  # 设定红色的低阈值
    high_range1 = np.array([180, 255, 255])  # 设定红色的高阈值

    mask1 = cv2.inRange(hsv_image, low_range, high_range)  # 根据阈值生成掩码
    mask2 = cv2.inRange(hsv_image, low_range1, high_range1)  # 根据阈值生成掩码

    mask = cv2.bitwise_or(mask1, mask2)
    print(mask)


if __name__ == '__main__':
    image = cv2.imread("pictures/img_2.png")
    seal_mask_handle(image)

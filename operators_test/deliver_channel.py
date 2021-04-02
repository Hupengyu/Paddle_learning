import cv2
import numpy as np
import os
import time

pwd = os.getcwd()
print(pwd)


def remove_red_seal(img):
    cv2.namedWindow('red_c', 0)
    cv2.resizeWindow('red_c', 700, 900)
    cv2.namedWindow('THRESH_OTSU', 0)
    cv2.namedWindow('THRESH_BINARY', 0)
    cv2.resizeWindow('THRESH_BINARY', 700, 900)
    cv2.namedWindow('expand_dims', 0)
    cv2.resizeWindow('expand_dims', 700, 900)
    cv2.namedWindow('result_img', 0)
    cv2.resizeWindow('result_img', 700, 900)

    # 分离图片的通道
    blue_c, green_c, red_c = cv2.split(img)
    cv2.imshow('red_c', red_c)
    cv2.waitKey()

    # 利用大津法自动选择阈值
    thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_OTSU)
    cv2.imshow('THRESH_OTSU', ret)
    cv2.waitKey()

    # 对阈值进行调整
    filter_condition = int(thresh * 0.98)
    # 移除红色的印章-----------所有大于该阈值的都转换为白色(255)
    # ******就是说THRESH_OTSU认为大于该thresh的“更像”红色******
    # 百分比越小红色印章移除的越干净---因为设定后的filter_condition越小
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
    cv2.imshow('THRESH_BINARY', red_thresh)
    cv2.waitKey()

    # 把图片转回3通道
    result_img = np.expand_dims(red_thresh, axis=2)
    cv2.imshow('expand_dims', result_img)
    cv2.waitKey()

    # result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
    result_img = cv2.merge(result_img, blue_c, green_c)
    cv2.imshow('result_img', result_img)
    cv2.waitKey()
    return result_img


time_0 = time.time()
img_path = os.path.join(pwd + "/pictures/" + "opencv.jpg")
res_path = os.path.join(pwd + "/results/" + "remove_seal.jpg")

input_img = cv2.imread(img_path)
time_read = time.time()
print(time_read-time_0)
remove_seal = remove_red_seal(input_img)
time_end = time.time()
print(time_end - time_read)
cv2.imwrite(res_path, remove_seal)

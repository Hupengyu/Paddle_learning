import cv2
import numpy as np
import os
import time

pwd = os.getcwd()
print(pwd)


def remove_red_seal(img):
    cv2.namedWindow('red_c', 0)
    cv2.resizeWindow('red_c', 700, 900)

    # 分离图片的通道
    blue_c, green_c, red_c = cv2.split(img)
    cv2.imshow('red_c', red_c)
    cv2.waitKey()

    result_img = np.concatenate((red_c, red_c, red_c), axis=-1)
    result_img = cv2.merge(result_img, blue_c, green_c)
    cv2.imshow('result_img', result_img)
    cv2.waitKey()
    return result_img


img_path = os.path.join(pwd + "/pictures/" + "opencv.jpg")
res_path = os.path.join(pwd + "/results/" + "remove_seal.jpg")

input_img = cv2.imread(img_path)
time_read = time.time()
remove_seal = remove_red_seal(input_img)
time_end = time.time()
cv2.imwrite(res_path, remove_seal)

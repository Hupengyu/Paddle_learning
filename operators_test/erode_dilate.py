import numpy as np
import cv2
import os

pwd = os.getcwd()
img_path = os.path.join(pwd + "/pictures/" + "lena.png")
res_path = os.path.join(pwd + "/results/" + "cut_images.jpg")
crops_path = os.path.join(pwd + "/results/")


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=1, fy=1)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def erode_dilate(img_path):
    img = cv2.imread(img_path)
    # show_img(img, 'img')

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    show_img(gray, 'gray')

    # 寻找边缘
    # edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    dilate1 = cv2.dilate(gray, kernel, iterations=1)  # 膨胀
    show_img(dilate1, 'dilate1')
    dilate2 = cv2.dilate(dilate1, kernel, iterations=1)  # 膨胀
    show_img(dilate2, 'dilate2')
    dilate3 = cv2.dilate(dilate2, kernel, iterations=1)  # 膨胀
    show_img(dilate3, 'dilate3')
    erode1 = cv2.erode(gray, kernel, iterations=1)  # 腐蚀
    show_img(erode1, 'erode1')
    erode2 = cv2.erode(erode1, kernel, iterations=1)  # 腐蚀
    show_img(erode2, 'erode2')
    erode3 = cv2.erode(erode2, kernel, iterations=1)  # 腐蚀
    show_img(erode3, 'erode3')
    # erode_dilate_res = cv2.erode(dilate3, kernel, iterations=3)  # 腐蚀
    # show_img(erode_dilate_res, 'erode_dilate_res')


if __name__ == '__main__':
    erode_dilate(img_path)

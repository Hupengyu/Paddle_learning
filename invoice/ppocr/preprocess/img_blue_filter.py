import cv2
import numpy as np


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


# 提取发票中蓝色和黑色的字体
def img_blue_filter(img):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 颜色空间转换

    # 将绿色区域(横竖线)和部分红色[0-15]排除
    low_0 = np.array([0, 30, 0])
    high_0 = np.array([67, 255, 255])

    low_1 = np.array([78, 30, 0])
    high_1 = np.array([124, 255, 255])
    # 将红色印章剔除
    low_2 = np.array([178, 30, 0])
    high_2 = np.array([255, 255, 255])

    mask1 = cv2.inRange(hsv_image, low_0, high_0)  # 根据阈值生成掩码
    mask2 = cv2.inRange(hsv_image, low_1, high_1)  # 根据阈值生成掩码
    mask3 = cv2.inRange(hsv_image, low_2, high_2)  # 根据阈值生成掩码

    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)

    content_mask = mask == 255  # 取mask中为255的设置为true

    blue_seal = np.zeros(img.shape, np.uint8)  # 新画布mask_white
    blue_seal[:, :] = (255, 255, 255)  # 喷白
    blue_seal[content_mask] = img[content_mask]  # (0,0,255)     # 利用red_mask将蓝色区域设置为白色

    # seal_mask_res = cv2.bitwise_and(img, img, mask=mask_white)  # 掩码掩盖后的图片(提取印章轮廓)
    # show_img(blue_seal, 'blue_seal')
    return blue_seal


if __name__ == '__main__':
    img_path = 'doc/imgs/000.jpg'
    img = cv2.imread(img_path)
    img_blue = img_blue_filter(img)
    print('aa')
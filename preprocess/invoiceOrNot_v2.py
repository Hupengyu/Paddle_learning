import cv2
import numpy as np
import os


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def detect_circle(image):
    # dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=5) # 高斯双边滤波(慢)
    # dst = cv2.pyrMeanShiftFiltering(image, 10, 100)  # 均值偏移滤波（稍微快）
    # dst = cv2.cvtColor(dst, cv2.COLOR_BGRA2GRAY)    # BGRA2GRAY
    seal_num = 0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # show_img(gray, 'gray')

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=5)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=3)  # 腐蚀
    # show_img(morphed, 'morphed')

    # 找轮廓
    morphed_copy = morphed.copy()
    # show_img(morphed_copy, 'morphed_copy')

    cnts, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    else:
        print("Did not find contours\n")
        return

    for box in cnts:
        # *********************轮廓点预处理***********************
        if len(box) < 200:
            continue
        epsilon = 0.01 * cv2.arcLength(box, True)   # 设定近似多边形的精度
        approx = cv2.approxPolyDP(box, epsilon, True)   # 根据精度重新绘制轮廓
        # ***************visual****************
        img_copy = image.copy()
        cv2.drawContours(img_copy, [approx], -1, (255, 0, 0), 2)
        approx_num = len(approx)
        # show_img(img_copy, 'approx_num:%f' % approx_num)
        if 8 < approx_num:    # 剔除噪音(少于10个点的轮廓剔除)
            # ******************ellipse识别*****************
            # ellipse = cv2.fitEllipse(approx)    # There should be at least 5 points
            # ellipse = cv2.ellipse(image, ellipse, (0, 255, 0), 4)
            # show_img(ellipse, 'ellipse')
            seal_num += 1
        else:
            continue
    return seal_num


def seal_mask_handle(img):
    # np.set_printoptions(threshold=np.inf)

    # img = cv2.resize(img, None, fx=0.5, fy=0.5,
    #                  interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 颜色空间转换
    # show_img(hsv_image, 'hsv_image')

    low_red0 = np.array([0, 45, 45])  # 设定红色的高阈值
    high_red0 = np.array([15, 119, 255])  # 设定红色的高阈值

    low_red1 = np.array([156, 100, 100])  # 设定红色的低阈值
    high_red1 = np.array([179, 255, 255])  # 设定红色的高阈值

    mask1 = cv2.inRange(hsv_image, low_red0, high_red0)  # 根据阈值生成掩码
    mask2 = cv2.inRange(hsv_image, low_red1, high_red1)  # 根据阈值生成掩码

    mask = cv2.bitwise_or(mask1, mask2)    # hsv的红色有两个范围

    red_mask = mask2 == 255  # 取mask中为255的设置为true

    red_seal = np.zeros(img.shape, np.uint8)  # 新画布mask_white
    red_seal[:, :] = (255, 255, 255)  # 喷白
    red_seal[red_mask] = img[red_mask]  # (0,0,255)     # 利用red_mask将红色区域设置为白色

    # seal_mask_res = cv2.bitwise_and(img, img, mask=mask_white)  # 掩码掩盖后的图片(提取印章轮廓)
    # show_img(red_seal, 'red_seal')
    return red_seal


def invoice_or_not(image):
    print('******************印章数量识别开始******************')
    # 先将红色印章mask
    seal_mask_res = seal_mask_handle(image)
    # 计数印章数量
    seal_num_res = detect_circle(seal_mask_res)
    print('seal_num_res: ', seal_num_res)
    # # 输出印章数量
    if seal_num_res % 2 != 0:
        print('识别失败')
    else:
        invoice_num = int(seal_num_res / 2)
        print('识别成功')
        print('invoice_num: ', invoice_num)
    print('******************印章数量识别结束******************')


if __name__ == '__main__':
    file_path = 'results/crops'
    # 读取函数，用来读取文件夹中的所有函数，输入参数是文件名
    for filename in os.listdir(file_path):
        img = cv2.imread(file_path + "/" + filename)
        invoice_or_not(img)

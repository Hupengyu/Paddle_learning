import cv2
import numpy as np
import os
from pdf2pages import pdf2pages
from preprocess import detect_image_counts
from cut_images import cut_images_save


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def detect_circle(image):
    seal_num = 0

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # show_img(gray, 'gray')

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((7, 7), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=5)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=3)  # 腐蚀

    morphed_copy = morphed.copy()
    # show_img(morphed_copy, 'morphed_copy')

    # 找外轮廓
    cnts, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    else:
        print("Did not find contours\n")
        return seal_num
    for box in cnts:
        if len(box) < 300 or len(box) > 800:
            continue
        print('box_nums: ', len(box))
        epsilon = 0.01 * cv2.arcLength(box, True)  # 设定近似多边形的精度
        approx = cv2.approxPolyDP(box, epsilon, True)  # 根据精度重新绘制轮廓
        # ***************visual****************
        img_copy = image.copy()
        cv2.drawContours(img_copy, [approx], -1, (255, 0, 0), 2)
        approx_num = len(approx)
        # show_img(img_copy, 'approx_num:%f' % approx_num)
        if 5 < approx_num:  # 剔除噪音(少于10个点的轮廓剔除)
            # ******************ellipse识别*****************
            # ellipse = cv2.fitEllipse(approx)    # There should be at least 5 points
            # ellipse = cv2.ellipse(image, ellipse, (0, 255, 0), 4)
            # show_img(ellipse, 'ellipse')
            seal_num += 1
        else:
            continue
    cv2.destroyAllWindows()
    return seal_num


def seal_mask_handle(img):
    # np.set_printoptions(threshold=np.inf)

    # img = cv2.resize(img, None, fx=0.5, fy=0.5,
    #                  interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）

    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 颜色空间转换
    # show_img(hsv_image, 'hsv_image')

    # low_red0 = np.array([0, 45, 45])  # 设定红色的高阈值
    # high_red0 = np.array([15, 119, 255])  # 设定红色的高阈值
    #
    # low_red1 = np.array([156, 100, 100])  # 设定红色的低阈值
    # high_red1 = np.array([179, 255, 255])  # 设定红色的高阈值

    low_red0 = np.array([0, 25, 194])  # 设定红色的高阈值
    high_red0 = np.array([15, 255, 255])  # 设定红色的高阈值

    low_red1 = np.array([124, 25, 194])  # 设定红色的低阈值
    high_red1 = np.array([179, 255, 255])  # 设定红色的高阈值

    mask1 = cv2.inRange(hsv_image, low_red0, high_red0)  # 根据阈值生成掩码
    mask2 = cv2.inRange(hsv_image, low_red1, high_red1)  # 根据阈值生成掩码

    mask = cv2.bitwise_or(mask1, mask2)  # hsv的红色有两个范围

    red_mask = mask2 == 255  # 取mask中为255的设置为true

    red_seal = np.zeros(img.shape, np.uint8)  # 新画布mask_white
    red_seal[:, :] = (255, 255, 255)  # 喷白
    red_seal[red_mask] = img[red_mask]  # (0,0,255)     # 利用red_mask将红色区域设置为白色

    # seal_mask_res = cv2.bitwise_and(img, img, mask=mask_white)  # 掩码掩盖后的图片(提取印章轮廓)
    # show_img(red_seal, 'red_seal')
    return red_seal


def invoice_or_not(image):
    global invoices_sum
    print('------------------印章数量识别开始------------------')
    # 先将红色印章mask
    seal_mask_res = seal_mask_handle(image)
    # 计数印章数量
    seal_num_res = detect_circle(seal_mask_res)
    print('seal_num_res: ', seal_num_res)
    # # 输出印章数量
    if seal_num_res == 0:
        print('识别失败')
        print('**********************印章数量识别结束***********************')
        return False
    else:
        print('识别成功')
        invoices_sum += 1
        print('**********************印章数量识别结束***********************')
        return True


if __name__ == '__main__':
    invoices_sum = 0
    pdf_path = './pictures/pdf/大别山14380采购价款.pdf'
    crops_save_path = './results/crops/'

    # ------pdf转images----- -
    if pdf_path[-3:] == 'pdf':
        imgs_list = pdf2pages(pdf_path)
    else:
        imgs_list = cv2.imread(pdf_path)

    # -----------处理每张图片，两张的话切割----------
    if type(imgs_list) == np.ndarray:
        invoices_num = detect_image_counts(imgs_list)
        if invoices_num > 1:
            cut_images_save(imgs_list, crops_save_path)
    else:
        for img in imgs_list:
            cut_images_save(img, crops_save_path)

    # -----------对切割后的图片进行Invoice判定----------
    for filename in os.listdir(crops_save_path):
        print('-------------------*****************************************---------------------')
        img = cv2.imread(crops_save_path + "/" + filename)
        print(crops_save_path + "/" + filename)
        invoice_or_not(img)

    print('发票总数： ', invoices_sum)

    # ------------单张测试------------
    # path = 'pictures/IMG_20191012_142702.jpg'
    # img = cv2.imread(path)
    # invoice_or_not(img)

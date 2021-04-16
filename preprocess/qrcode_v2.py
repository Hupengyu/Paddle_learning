# coding=utf-8
from __future__ import print_function
import numpy as np
import cv2
import fitz
from log_config import log
import pyzbar.pyzbar as pyzbar

logging = log()


def format_data(data):
    res = {}
    list_1 = data.split(',')
    res['发票代码'] = list_1[2]
    res['发票号码'] = list_1[3]
    # 20190712 -> 2019年07月12日
    a_list = list(list_1[5])[:8]
    a_list.insert(4, '-')
    a_list.insert(7, '-')
    resp = ''.join(a_list)
    res['开票日期'] = resp
    res['税后价格'] = list_1[4]
    res['校验码'] = list_1[6]

    return res


def show_img(img, win_name):
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


# ---------------------------二维码模块---------------------------
def filter_blue(image):
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # 阈值内的设为255,其余为0
    # res = cv2.bitwise_and(image, image, mask=mask)
    blue_mask = mask == 255  # 取mask中为255的设置为true

    blue_areas = np.zeros(image.shape, np.uint8)  # 创建新画布
    blue_areas[:, :] = (255, 255, 255)  # 画布喷白
    blue_areas[blue_mask] = image[blue_mask]  # 将blue的像素点‘喷’到白色画布上
    # 返回Img中的蓝色像素点
    return blue_areas


def get_contours(img0):
    gray = cv2.cvtColor(img0, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((5, 5), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=3)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=3)  # 腐蚀

    morphed_copy = morphed.copy()
    contours, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours, gray


def pick_rectangels(contours, gray_image):
    boxs = []
    print('count_contours: ', len(contours))
    for contour in contours:
        if len(contour) < 100 or len(contour) > 800:
            continue
        rect = cv2.minAreaRect(contour)     # 获取最小外接矩阵，中心点坐标，宽高，旋转角度
        max_len = max(rect[1][0], rect[1][1])
        min_len = min(rect[1][1], rect[1][0])
        ratio_w_h = max_len / min_len
        if ratio_w_h > 1.5 and min_len < 50:
            continue

        print('ratio_w_h: ', ratio_w_h)
        print(len(contour))

        box = np.int0(cv2.boxPoints(rect))  # 获取矩形四个顶点，浮点型[(tr, tl, bl, br) = src_coor]
        if box.shape[0] is not 4:
            continue
        boxs.append(box)

        img_copy = gray_image.copy()
        cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)   # 只需要四个点
        show_img(img_copy, 'drawContours')

    return boxs


def decode_qrcodes(boxs, original_image):
    for box in boxs:  # 对识别到的每一个正方形进行二维码识别，返回信息则确定
        x1 = box[1][0]-5
        x2 = box[3][0]+5
        y1 = box[1][1]-5
        y2 = box[3][1]+5

        original_image_copy = original_image.copy()
        cv2.rectangle(original_image_copy, (x1, y1), (x2, y2), (0, 0, 255), 4)
        show_img(original_image_copy, 'decode_qrcodes')

        qrcode_cute = original_image[y1:y2, x1:x2]  # y1:y2, x1:x2
        try:
            show_img(qrcode_cute, 'qrcode_cute')
            message = decode_qrcode(qrcode_cute)
        except Exception:
            # print('没有图片')
            continue
        if message is not None:
            return message, qrcode_cute
    return None, None


def decode_qrcode(image):
    barcodes = pyzbar.decode(image)
    # for barcode in barcodes:
    barcodeData = barcodes.data.decode("utf-8")
    print('barcodeData: ', barcodeData)
    return barcodeData
    # try:
    #     return barcodeData
    # except:
    #     return None


# 黑色二维码识别工具方法
def qrcode_common(filter_blue_image, original_image):
    print('进入黑色二维码识别区')
    cnts, gray = get_contours(filter_blue_image)  # 得到所有的外轮廓
    boxs = pick_rectangels(cnts, filter_blue_image)  # 获取到近似正方形boxs
    message, image = decode_qrcodes(boxs, gray)  # 二维码识别box
    return message


# 蓝色二维码识别工具方法
def qrcode_blue(image):
    print('进入蓝色二维码识别区')
    # 先把颜色过滤，再获取边框
    filter_blue_image = filter_blue(image)
    message = qrcode_common(filter_blue_image, image)
    return message


def qrcode(image):
    # 蓝色二维码识别
    qrcode_blue_info = qrcode_blue(image)
    if qrcode_blue_info is None:
        logging.info('二维码识别失败1')
        return qrcode_blue_info
    else:
        try:
            res = format_data(qrcode_blue_info)
        except IndexError:
            logging.info('二维码识别失败2')
        else:
            logging.info("二维码识别成功: " + str(res))
            return res


if __name__ == '__main__':
    # 读取图片
    img_path = './pictures/Image_00137.jpg'
    img = cv2.imread(img_path)
    qrcode(img)

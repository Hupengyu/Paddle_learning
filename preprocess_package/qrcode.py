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
    morphed = cv2.dilate(edged, kernel, iterations=5)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=5)  # 腐蚀

    morphed_copy = morphed.copy()
    contours, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    return contours, gray


def pick_rectangels(contours, gray_image):
    boxs = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # minAreaRect
        if (abs(w - h) < 20) & (w > 50):  # 找近似正方形
            boxs.append([x, y, w, h])
            if len(contour) < 300 or len(contour) > 1200:
                continue
            print('x, y, w, h: ', x, y, w, h)
            # epsilon = 0.01 * cv2.arcLength(contour, True)  # 设定近似多边形的精度
            # approx = cv2.approxPolyDP(contour, epsilon, True)  # 根据精度重新绘制轮廓
            img_copy = gray_image.copy()
            # # rect = cv2.minAreaRect(approx)
            # # box = np.int0(cv2.boxPoints(rect))
            # cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 5)
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 3)
            show_img(img_copy, 'boxs')
        else:
            continue

    return boxs


def decode_qrcodes(boxs, original_image):
    for box in boxs:  # 对识别到的每一个正方形进行二维码识别，返回信息则确定
        x1 = box[0]             # 左x坐标
        x2 = box[0] + box[2]    # 右x坐标
        y1 = box[1]             # 上y坐标
        y2 = box[1] + box[3]    # 下y坐标

        original_image_copy = original_image.copy()
        cv2.rectangle(original_image_copy, (x1, y1), (x2, y2), (255, 0, 0), 4)  # 对角坐标
        show_img(original_image, 'decode_qrcodes')

        qrcode_cute = original_image[y1:y2, x1:x2]  # y1:y2, x1:x2
        show_img(qrcode_cute, 'qrcode_cute')

        message = decode_qrcode(qrcode_cute)

        if message != -1:
            return message, qrcode_cute
    return None, None


def decode_qrcode(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
    try:
        return barcodeData
    except:
        return -1


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
        logging.info('二维码识别失败')
        return qrcode_blue_info
    else:
        res = format_data(qrcode_blue_info)
        logging.info("二维码识别成功: " + str(res))
        return res


if __name__ == '__main__':
    # 读取图片
    img_path = './pictures/Image_00036.jpg'
    img = cv2.imread(img_path)
    qrcode(img)

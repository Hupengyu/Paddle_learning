import numpy as np
import cv2
import utils
from log_config import log

logging = log()

# 蓝色的范围，不同光照条件下不一样，可灵活调整
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])


# lower_blue = np.array([0, 0, 0])
# upper_blue = np.array([180, 255, 46])


def detection_outline(image):
    # image = cv2.imread('invoice_2.jpg')
    # image = cv2.imread('in_002.jpg')
    # image = cv2.imread('in_3.jpg')  # 黑色
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv', hsv)
    # 3.inRange()：介于lower/upper之间的为白色，其余黑色
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # cv2.imshow('mask', mask)
    # 4.只保留原图中的蓝色部分
    res = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    # cv2.imshow("gradient",gradient)
    # 原本没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
    (_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh",thresh)
    #
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed",closed)
    #
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # img, cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    cv2.drawContours(image, [box], -1, (0, 255, 0), 1)

    y_min, y_max, x_min, x_max = utils.get_max_np(box)

    image = image[y_min - 30:y_max + 30, x_min - 30:x_max + 30]
    # image = image[y_min:y_max, x_min:x_max]
    # cv2.imshow("image",image)
    # cv2.imwrite('image/image.jpg',image)
    #  print("------------------------------")
    # binary = recognition(only_qrcode)
    # decode(binary)
    return image


def recognition(image):
    # image = cv2.imread('image/image.jpg')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    res = cv2.bitwise_and(image, image, mask=mask)
    # 将二值化图形成闭空间
    kernel = np.ones((30, 30), np.uint8)
    close_image = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    qr_cnt = utils.ser(close_image)
    # 通过多边形拟合找到当前二维码轮廓的多边形，并对其描边 2px
    epsilon = 1
    approx = cv2.approxPolyDP(qr_cnt, epsilon, True)
    poly_image = np.ones(close_image.shape, np.uint8)
    cv2.polylines(poly_image, [approx], True, (255, 255, 255), 2)

    # 对这张黑色背景的图片取反，获得对应的白色图片
    bit_not_poly_image = cv2.bitwise_not(poly_image)

    # 递归填充黑色多边形
    fill_image = utils.fill_black_poly(bit_not_poly_image)

    # 将填充完成的图片按位取反并进行二值化处理，作为截取二维码区域的蒙版
    qr_mask_image = cv2.bitwise_not(fill_image)
    ret, qr_mask = cv2.threshold(qr_mask_image, 175, 255, cv2.THRESH_BINARY)

    # 通过蒙版截取图片二维码部分
    cut_image = cv2.bitwise_and(image, image, mask=qr_mask)

    # 将黑色部分修改为白色，方便后续处理
    cut_image = np.where(cut_image == 0, 255, cut_image)

    # 图片预处理，旋转方正并拉伸到407 * 407分辨率
    image = utils.prepare_image(cut_image, qr_cnt)
    # 去除多余颜色的线条干扰
    gray = utils.remove_other_line(image)
    # gray = utils.improvement_remove_other_line(image)
    # cv2.imshow("gray",gray)
    # 自适应二值化
    strecthed_blue_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 71, -10)
    # cv2.imshow("strecthed_blue_image",strecthed_blue_image)

    return strecthed_blue_image


def qrcode(image):
    try:
        # 蓝色二维码识别
        only_qrcode = detection_outline(image)  # 检测外边缘——这里很重要
        binary = recognition(only_qrcode)   # 识别
        data = utils.decode_original(binary)[1]    # 转换格式
        if len(data) == 0:
            logging.info('蓝色二维码识别失败 开始ocr文字识别')
        else:
            res = utils.format_data(data)
            logging.info("蓝色二维码识别成功: " + str(res))
    except Exception:
        data = utils.decode(image)
        if len(data) == 0:
            logging.info('黑色二维码识别失败 开始ocr文字识别')
        else:
            res = utils.format_data(data)
            logging.info('黑色二维码识别成功: ' + str(res))


if __name__ == '__main__':
    # 读取图片
    img_path = 'pictures/1.png'
    img = cv2.imread(img_path)
    qrcode(img)

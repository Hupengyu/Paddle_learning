import cv2
import copy
import numpy as np
import pyzbar.pyzbar as pyzbar


def prethreatment(img0):
    # read img and copy

    img = copy.deepcopy(img0)
    ##cv2.imshow('img',img)

    # make img into gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ##cv2.imshow('gray',gray)

    # threshold
    ret, thre = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    ##cv2.imshow('thre',thre)

    # erode
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(thre, kernel)
    erosion = cv2.erode(erosion, kernel)

    # cv2.imshow('erosion',erosion)

    # findContours
    contours, hier = cv2.findContours(erosion,
                                            cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_SIMPLE)
    return contours, gray


def pick_rectangels(contours):
    # choosecontours
    rec = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)  # 计算出一个简单地边界框
        if (abs(w - h) < 10) & (w > 50):
            rec.append([x, y, w, h])
    # print(rec)
    return rec


def decode_qrcodes(rec, gray):
    for r in rec:
        x1 = r[0]
        x2 = r[0] + r[2]
        y1 = r[1]
        y2 = r[1] + r[3]
        # print(x1, x2, y1, y2)

        img = gray[y1:y2, x1:x2]
        # 放大一点
        img = cv2.resize(img, (r[3] * 2, r[2] * 2))

        message = decode_qrcode(img)

        if message != -1:
            return message, img


def decode_qrcode(image):
    barcodes = pyzbar.decode(image)
    for barcode in barcodes:
        # 条形码数据为字节对象，所以如果我们想在输出图像上
        # 画出来，就需要先将它转换成字符串
        barcodeData = barcode.data.decode("utf-8")
        barcodeType = barcode.type
        # 提取条形码的边界框的位置
        # 画出图像中条形码的边界框
        # (x, y, w, h) = barcode.rect
        # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 向终端打印条形码数据和条形码类型
        # print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
    try:
        return barcodeData
    except:
        return -1


if __name__ == '__main__':
    imgpath = "8b8b8cd9a12ba471ba5fd3102d6b7af.jpg"
    img0 = cv2.imread(imgpath)
    # img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    contours, gray = prethreatment(img0)
    rec = pick_rectangels(contours)

    message, img = decode_qrcodes(rec, gray)
    print(message)

    # cv2.imshow('0', img0)
    # cv2.imshow('1', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

import cv2
import pyzbar.pyzbar as pyzbar
import numpy as np


def show_img(img, win_name):
    # img = cv2.resize(img, None, fx=0.5, fy=0.5)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def decode_qrcode(image):
    barcodes = pyzbar.decode(image)
    # for barcode in barcodes:
    if len(barcodes) == 0:
        print('barcodes is None')
        return None
    print('barcodes: ', barcodes)
    barcodeData = barcodes.data.decode("utf-8")
    print('barcodeData: ', barcodeData)
    return barcodeData


def qrcode_test(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    show_img(gray, 'gray')
    # 高斯模糊，消除一些噪声
    hist_gray = cv2.equalizeHist(gray)
    show_img(hist_gray, 'hist_gray')
    #
    _, thre_hist_gray = cv2.threshold(hist_gray, 160, 255, cv2.THRESH_BINARY)
    show_img(thre_hist_gray, 'thre_hist_gray')

    blur_gray = cv2.GaussianBlur(thre_hist_gray, (5, 1), 0)   # kernel设置为X方向的矩形，二维码横向断的情况多
    show_img(blur_gray, 'blur_gray')

    # scale_gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    # scale_gray = cv2.normalize(img, dst=None, alpha=350, beta=10, norm_type=cv2.NORM_MINMAX)
    # show_img(scale_gray, 'scale_gray')
    # edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # dst = cv2.adaptiveThreshold(gray, 160, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
    _, thre_gray = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    show_img(thre_gray, 'thre_gray')

    # kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    # morphed = cv2.dilate(gray, kernel, iterations=2)  # 膨胀
    # morphed = cv2.erode(dst, kernel, iterations=1)  # 腐蚀
    # show_img(dst, 'dst')
    # QRCODE
    message = decode_qrcode(blur_gray)
    if message is None:
        message = decode_qrcode(thre_gray)
    if message is None:
        message = decode_qrcode(thre_hist_gray)

    return message


if __name__ == '__main__':
    # 读取图片
    img_path = './pictures/qrcode.jpg'
    img = cv2.imread(img_path)
    res = qrcode_test(img)
    print('识别结果： ', res)

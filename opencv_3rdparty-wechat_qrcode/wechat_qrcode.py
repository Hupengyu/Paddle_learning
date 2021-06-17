import cv2
import os
import numpy as np

model_path = 'opencv_3rdparty-wechat_qrcode'
imgs_path = 'qrcode_imgs'


def wechat_qrcode(image):
    img_strong = image_enhancement(image)
    detector = cv2.wechat_qrcode_WeChatQRCode(model_path + "/" + "detect.prototxt", model_path + "/" +"detect.caffemodel", model_path + "/" +"sr.prototxt", model_path + "/" +"sr.caffemodel")
    res_qrcode, _ = detector.detectAndDecode(img_strong)
    return res_qrcode, _


def show_img(img, win_name, ratio=1.0):
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def image_enhancement(img):
    def gamma(img, c, v):
        lut = np.zeros(256, dtype=np.float32)
        for i in range(256):
            lut[i] = c * i ** v
        output_img = cv2.LUT(img, lut)
        output_img = np.uint8(output_img + 0.5)  # 这句一定要加上
        return output_img

    out2 = gamma(img, 0.00000005, 4.0)
    show_img(out2, 'out2')
    # 直方图均衡增强-pass
    # result = img.copy()
    # for j in range(3):
    #     result[:, :, j] = cv2.equalizeHist(img[:, :, j])
    # # cv2.imshow('Result1', result)
    # show_img(result, 'result')

    # 拉普拉斯算子增强-pass
    # kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])  # 定义卷积核
    # imageEnhance = cv2.filter2D(img, -1, kernel)  # 进行卷积运算
    # show_img(imageEnhance, 'result')

    # 灰度锐化-pass
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # # 高斯模糊，消除一些噪声
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # # show_img(gray, 'gray')
    #
    # # 寻找边缘
    # edged = cv2.Canny(gray, 50, 120)
    # # show_img(edged, 'edged')
    #
    # # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    # kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    # morphed = cv2.dilate(edged, kernel, iterations=1)  # 膨胀
    # morphed = cv2.erode(morphed, kernel, iterations=1)  # 腐蚀
    # show_img(morphed, 'morphed')

    return out2


if __name__ == '__main__':
    for file in os.listdir(imgs_path):
        file_name = imgs_path + "/" + file
        images = cv2.imread(file_name)
        res, _ = wechat_qrcode(images)
        print(file_name)
        print(res)

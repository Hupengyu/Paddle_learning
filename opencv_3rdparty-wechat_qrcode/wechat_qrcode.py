import cv2
import os

model_path = 'opencv_3rdparty-wechat_qrcode'
imgs_path = 'qrcode_imgs'

detector = cv2.wechat_qrcode_WeChatQRCode(model_path + "/" + "detect.prototxt", model_path + "/" +"detect.caffemodel", model_path + "/" +"sr.prototxt", model_path + "/" +"sr.caffemodel")

for file in os.listdir(imgs_path):
    file_name = imgs_path + "/" + file
    img = cv2.imread(file_name)
    res, points = detector.detectAndDecode(img)
    print(file_name)
    print(res, points)
import cv2

detector = cv2.wechat_qrcode_WeChatQRCode("opencv_3rdparty-wechat_qrcode/detect.prototxt", "opencv_3rdparty"
                                                                                           "-wechat_qrcode/detect"
                                                                                           ".caffemodel",
                                          "opencv_3rdparty-wechat_qrcode/sr.prototxt", "opencv_3rdparty-wechat_qrcode"
                                                                                       "/sr.caffemodel")
img = cv2.imread("pictures/qrcode.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

res, points = detector.detectAndDecode(img)
print(res)


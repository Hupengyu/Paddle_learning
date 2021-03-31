import base64
import numpy as np
import cv2
import os
from PIL import Image
from io import BytesIO

pwd = os.getcwd()
img_path = os.path.join(pwd + "/pictures/" + "aaaa.jpg")
res_path = os.path.join(pwd + "/results/" + "cut_images.jpg")
crops_path = os.path.join(pwd + "/results/")

img = cv2.imread(img_path)  # narray
img_encode = cv2.imencode('.jpg', img)[1]
image_code = str(base64.b64encode(img_encode))[2:-1]
# base64.b64encode(img_encode)
# image_code = img_encode.tostring()
# f = open(r'{}'.format(img_path), 'rb')
# image = base64.b64encode(img)
# img_b64encode = img.tobytes()     # base64编码

img_b64decode = base64.b64decode(image_code)

img_array = np.fromstring(img_b64decode, np.uint8)  # 转换np序列
img_2 = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)  # 转换Opencv格式

cv2.imshow("img", img_2)

cv2.waitKey()
print('aaa')

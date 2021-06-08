import base64
import numpy as np
import cv2
import os

pwd = os.getcwd()
img_path = os.path.join(pwd + "/pictures/" + "xigua.png")
res_path = os.path.join(pwd + "/results/" + "xigua_flow.png")
crops_path = os.path.join(pwd + "/results/")

img = cv2.imread(img_path)  # cv2读取图片:ndarray:(500, 500, 3)
img_cv2_encode = cv2.imencode('.png', img)[1]   # 将图片编码成流数据:{ndarray:(61745, 1)}

# b64encode是为了加密。好处是编码后的文本数据可以在邮件正文、网页等直接显示。而且base64特别适合在http，mime协议下快速传输数据。
img_b64encode = base64.b64encode(img_cv2_encode)    # b64编码：{bytes: 82328}

img_b64decode = base64.b64decode(img_b64encode)    # b64解码:{bytes: 61745}

img_array = np.fromstring(img_b64decode, np.uint8)  # 将str解码成流数据:{ndarray:(61745, )}
img_2 = cv2.imdecode(img_array, cv2.COLOR_RGB2BGR)  # cv2解码流数据:ndarray:(500, 500, 3)

cv2.imshow("img", img_2)
cv2.imwrite(res_path, img_2)
cv2.waitKey()
print('aaa')

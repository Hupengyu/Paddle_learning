import cv2
import numpy as np
from PIL import Image

# 图片文件路径
path = 'pictures/xigua.png'
# 用cv2打开图片
img_cv2 = cv2.imread(path)
print(type(img_cv2))
# 用PIL打开文件
img_pil = Image.open(path)
print(type(img_pil))
# 将读取的图片转换为数组
img_pil = np.asarray(img_pil)
# 得到r，g，b三个分量
# r, g, b = cv2.split(img_pil)
# # 以b，g，r分量重新生成新图像
# img_pil = cv2.merge([b, g, r])
# 显示图片
cv2.imshow("cv2", img_cv2)
cv2.imshow("pil", img_pil)
cv2.waitKey()

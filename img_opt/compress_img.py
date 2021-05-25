from PIL import Image
import cv2
import os

file = 'file/001_270.jpg'
# img = Image.open(file)
# w, h = img.size
# w, h = round(w), round(h)  # 去掉浮点，防报错
img = cv2.imread(file)
img_size = os.path.getsize(file)/1024/1024
img_height, img_width = img.shape[0:2]

while img_size > 1:
    # 压缩图片
    img.thumbnail((int(img_width/2), int(img_height/2)), Image.ANTIALIAS)
    img.save('results/1.jpg','jpg')  # 质量
    img_size = os.path.getsize(file) / 1024 / 1024

cv2.imwrite('results/1.jpg', img)
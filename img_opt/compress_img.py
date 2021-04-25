from PIL import Image
import cv2

file = 'pictures/changshu01.png'
# img = Image.open(file)
# w, h = img.size
# w, h = round(w), round(h)  # 去掉浮点，防报错
img = cv2.imread(file)
img_height, img_width = img.shape[0:2]

img = cv2.resize(img, (img_width, img_height), cv2.INTER_LANCZOS4)
# img.save('results/1.jpg','jpg')  # 质量
cv2.imwrite('results/1.jpg', img)
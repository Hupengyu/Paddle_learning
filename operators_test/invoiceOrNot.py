# 去除印章
import cv2
import numpy as np
import matplotlib.pyplot as plt

image0 = cv2.imread("pictures/hsv_color.png", cv2.IMREAD_COLOR)  # 以BGR色彩读取图片
image = cv2.resize(image0, None, fx=0.8, fy=0.8,
                   interpolation=cv2.INTER_CUBIC)  # 缩小图片0.5倍（图片太大了）
cols, rows, _ = image.shape  # 获取图片高宽
B_channel, G_channel, R_channel = cv2.split(image)  # 注意cv2.split()返回通道顺序

cv2.imshow('Blue channel', B_channel)
cv2.imshow('Green channel', G_channel)
cv2.imshow('Red channel', R_channel)

pixelSequence = R_channel.reshape([rows * cols, ])  # 红色通道的histgram 变换成一维向量
numberBins = 256  # 统计直方图的组数
plt.figure()  # 计算直方图
manager = plt.get_current_fig_manager()
histogram, bins, patch = plt.hist(pixelSequence,
                                  numberBins,
                                  facecolor='black',
                                  histtype='bar')  # facecolor设置为黑色
# 设置坐标范围
y_maxValue = np.max(histogram)
plt.axis([0, 255, 0, y_maxValue])
# 设置坐标轴
plt.xlabel("gray Level", fontsize=20)
plt.ylabel('number of pixels', fontsize=20)
plt.title("Histgram of red channel", fontsize=25)
plt.xticks(range(0, 255, 10))
# 显示直方图
# plt.pause(0.05)
plt.ioff()
plt.savefig("histgram.png", dpi=260, bbox_inches="tight")
plt.show()

# 红色通道阈值(调节好函数阈值为160时效果最好，太大一片白，太小干扰点太多)
_, RedThresh = cv2.threshold(R_channel, 160, 255, cv2.THRESH_BINARY)

# 膨胀操作（可以省略）
element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
erode = cv2.erode(RedThresh, element)

# 显示效果
cv2.imshow('original color image', image)
cv2.imshow("RedThresh", RedThresh)
cv2.imshow("erode", erode)

# 保存图像
cv2.imwrite('scale_image.jpg', image)
cv2.imwrite('RedThresh.jpg', RedThresh)
cv2.imwrite("erode.jpg", erode)

cv2.waitKey(0)
cv2.destroyAllWindows()

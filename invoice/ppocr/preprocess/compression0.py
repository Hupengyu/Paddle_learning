import cv2
import os
from PIL import Image
from PIL import ImageEnhance
from PIL import ImageFilter
import cv2

if __name__ == '__main__':

    infile_dir = os.path.join(os.getcwd(), "doc/pp/", "IMG_20191012_142414.jpg")
    outfile_dir1 = os.path.join(os.getcwd(), "preprocess_result_imgs/", "compress_gray1.jpg")
    outfile_dir2 = os.path.join(os.getcwd(), "preprocess_result_imgs/", "compress_gray2.jpg")
    outfile_dir3 = os.path.join(os.getcwd(), "preprocess_result_imgs/", "compress_gray3.jpg")

    image = cv2.imread(infile_dir)
    res = cv2.resize(image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
    imgE = Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
    imgEH = ImageEnhance.Contrast(imgE)
    # 当参数为1.2 灰度图243KB，当参数为2.8 灰度图124KB.（亮度提升后转灰度图，图片会黑白分化）
    gray = imgEH.enhance(11.2).convert("L")
    gray.save(outfile_dir1)

    # 图像增强
    # 创建滤波器，使用不同的卷积核
    gary2 = gray.filter(ImageFilter.DETAIL)
    gary2.save(outfile_dir2)

    # 图像点运算
    gary3 = gary2.point(lambda i: i * 0.9)
    gary3.save(outfile_dir3)

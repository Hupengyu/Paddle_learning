import cv2
from PIL import Image
import os
import sys


def show_img(img, win_name, ratio=0.3):
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def get_img_size(img):
    sp = img.shape
    print('common_size: ', (sp[0] * sp[1] * sp[2])/1024/1024/sp[2])


def compress_img(image):
    img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # img = cv2.imread(img_path)
    img_width, img_height = img.size
    print('os_size: %s, %s ' % (img_height, img_width))
    # img_size = img.size/1024/1024
    os_size = os.path.getsize(img_path)     # 返回字节数
    img_size = os_size / 1024 / 1024         # 单位换算为M
    while img_size > 1:
        img.thumbnail((int(img_width/2), int(img_height/2)), Image.ANTIALIAS)
        img.save(img_path)
        img = Image.open(img_path)
        img_width, img_height = img.size
        print('img_shape: %s, %s ' % (img_height, img_width))
        # img.resize((int(img_width / 2), int(img_height / 2)))
        img_size = os.path.getsize(img_path) / 1024 / 1024    # 返回字节数
        print('img_size: ', img_size)
    img_compressed = cv2.imread(img_path)

    return img_compressed


if __name__ == '__main__':
    img_path = 'file/01.jpg'
    img = cv2.imread(img_path)
    img_compressed = compress_img(img)
    show_img(img_compressed, 'show_img')

    # img = cv2.imread(img_path)
    # get_img_size(img)

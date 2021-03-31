from pdf2png_v2 import pdf2image
from preprocess import detect_image_counts, cut_image
import os
import cv2
import numpy

pwd = os.getcwd()

image_index = 1


def cut_images(img):
    global image_index
    print('**********************开始处理图片*************************')
    wrap = detect_image_counts(img)
    print('wrap: ', wrap)
    if wrap == 2:
        print('-------------此图片有两张发票----------------')
        imgs_list = cut_image(img)
        for img in imgs_list:
            cv2.imwrite(crops_save_path + str(image_index) + '.png', img)
            image_index += 1
    else:
        cv2.imwrite(crops_save_path + str(image_index) + '.png', img)
        image_index += 1
    print('***************************************处理图片完成*************************************')


if __name__ == '__main__':
    file_path = './pictures/img_2.png'
    crops_save_path = './results/crops/'

    # ------pdf转images------
    if file_path[-3:] == 'pdf':
        imgs_list = pdf2image(file_path)
    else:
        imgs_list = cv2.imread(file_path)

    # -----------处理图片开始----------
    if type(imgs_list) == numpy.ndarray:
        invoices_num = detect_image_counts(imgs_list)
        if invoices_num > 1:
            cut_images(imgs_list)
    else:
        for img in imgs_list:
            cut_images(img)


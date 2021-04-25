from preprocess_package.pdf2pages import pdf2pages
from preprocess_package.preprocess import detect_image_counts
from preprocess_package.cut_images import cut_images_save
import os
import cv2
import numpy

pwd = os.getcwd()

image_index = 1

if __name__ == '__main__':
    imgs_file_path = './pictures/image'
    pdf_file_path = './pictures/pdf'
    img_path = './pictures/image/imgs.jpg'
    crops_save_path = './results/crops/'

    # ------pdf转images------
    if pdf_file_path[-3:] == 'pdf':
        imgs_list = pdf2pages(pdf_file_path)
    else:   # 单张图片
        imgs_list = cv2.imread(img_path)

    # -----------单张图片----------
    if type(imgs_list) == numpy.ndarray:
        invoices_num = detect_image_counts(imgs_list)
        if invoices_num > 1:
            cut_images_save(imgs_list, crops_save_path)

    # -----------多张图片----------
    else:
        for img in imgs_list:                           # pdf文件夹
            cut_images_save(img, crops_save_path)
        for img_name in os.listdir(imgs_file_path):     # 图片文件夹
            img = imgs_file_path + "/" + img_name
            img = cv2.imread(img)
            cut_images_save(img, crops_save_path)
from pdf2pages import pdf2pages
from preprocess import detect_image_counts
from preprocess.cut_images import cut_images_save
import os
import cv2
import numpy

pwd = os.getcwd()

image_index = 1

if __name__ == '__main__':
    file_path = './pictures/01407883-90发票.pdf'
    crops_save_path = './results/crops/'

    # ------pdf转images------
    if file_path[-3:] == 'pdf':
        imgs_list = pdf2pages(file_path)
    else:
        imgs_list = cv2.imread(file_path)

    # -----------处理图片开始----------
    if type(imgs_list) == numpy.ndarray:
        invoices_num = detect_image_counts(imgs_list)
        if invoices_num > 1:
            cut_images_save(imgs_list)
    else:
        for img in imgs_list:
            cut_images_save(img)
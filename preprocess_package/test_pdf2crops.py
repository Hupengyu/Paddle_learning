from preprocess_package.pdf2pages import pdf2pages
from preprocess_package.cut_images import cut_images_save
import os
import cv2

pwd = os.getcwd()

image_index = 1

if __name__ == '__main__':
    imgs_file_path = './pictures/发票扫描图片'
    failed_imgs_file_path = './pictures/failed_images'
    single_img_file_path = './pictures/single_image/'
    pdf_file_path = './pictures/pdf/山东宏瑞达开票4.28.pdf'
    img_path = './pictures/image/Image_00096.jpg'
    crops_save_path = './results/crops/'

    path_now = pdf_file_path

    # # # ------pdf转images------
    if path_now[-3:] == 'pdf':
        imgs_list = pdf2pages(path_now)
        for img in imgs_list:  # pdf文件夹
            cut_images_save(img=img, if_show_pre=False, if_show=False, img_name='', save_path='./results/crops/')
    else:  # 单张图片
        for img_name in os.listdir(path_now):
            img = path_now + "/" + img_name
            img = cv2.imread(img)
            cut_images_save(img=img, if_show_pre=False, if_show=False, img_name='', save_path='./results/crops/')
    #
    # # -----------单张图片----------
    # if type(imgs_list) == numpy.ndarray:
    #     invoices_num = detect_image_counts(imgs_list)
    # if invoices_num > 1:
    #     cut_images_save(imgs_list, crops_save_path)

    # -----------多张图片----------
    # else:
    # imgs_list = cv2.imread(imgs_file_path)

    # path = imgs_file_path
    #
    # global if_show_pre
    # if_show_pre = False
    # for img_name in os.listdir(crops_save_path):     # 图片文件夹
    #     img = crops_save_path + "/" + img_name
    #     img = cv2.imread(img)
    #     cut_images_save(img, if_show_pre, img_name, crops_save_path)

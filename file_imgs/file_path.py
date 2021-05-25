import os
import uuid
import shutil
import cv2


def initialize_images_dir(imagesPath):
    '''
    初始化 存放图片需要的 文件夹
    :return: void
    '''
    # imagesPath = "temporary_img"
    if not os.path.exists(imagesPath):
        os.makedirs(imagesPath)


if __name__ == '__main__':
    invoice_file_name = 'histgram.png'
    img = cv2.imread('histgram.png')

    upload_path = "temporary_img"
    initialize_images_dir(upload_path)
    url_path = os.path.join(upload_path, str(uuid.uuid1()))
    initialize_images_dir(url_path)
    images_path = os.path.join(url_path, invoice_file_name)
    cv2.imwrite(images_path, img)
    if os.path.exists(url_path):
        shutil.rmtree(url_path)
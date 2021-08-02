import base64
import os

import cv2
import numpy as np
import filetype
import tempfile

from ppocr.utils.logging import get_logger

logging = get_logger()


# image_path:文件绝对路径
def if_img_type(image_path):
    kind = filetype.guess(image_path)
    if kind is None:
        logging.info('Cannot guess file type!')
        return

    newpath = image_path + '.' + kind.extension

    os.rename(image_path, newpath)
    # logging.info('kind.extension: %s', kind.extension)

    return kind.extension


if __name__ == '__main__':
    image_path = '../../test/d3e7f531-a366-4f1e-99f0-c2fc4e8c7862'
    # images_save__path = '../test/a.png'
    # img = cv2.imread(image_path)  # 读取文件
    # image_baidu_encode = cv2.imencode('.jpg', img)[1]
    # image_baidu_encode = base64.b64encode(image_baidu_encode)  # bytes

    img_extension = if_img_type(image_path)
    if img_extension is None:
        print('此文件并不是图片或pdf')
    else:
        print('img_extension: ', img_extension)

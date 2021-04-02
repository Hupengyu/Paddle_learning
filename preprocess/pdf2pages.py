import fitz
import cv2
import os
import numpy


def pdf2images(pdf_name):
    pdf = fitz.Document(pdf_name)
    imgs_list = []
    for i, pg in enumerate(range(0, pdf.pageCount)):
        page = pdf[pg]  # 获得每一页的对象
        trans = fitz.Matrix(2.0, 2.0)
        pm = page.getPixmap(matrix=trans, alpha=False)  # 获得每一页的流对象
        img_path = pdf_name[:-4] + '_' + str(pg + 1) + '.png'
        pm.writePNG(img_path)  # 保存图片
        img = cv2.imread(img_path)
        imgs_list.append(img)
        os.remove(img_path)  # 删除图片
    pdf.close()
    return imgs_list


if __name__ == '__main__':
    pdf_path = './pictures/pdf/发票3.pdf'
    crops_save_path = './results/crops/'

    image_index = 1

    # ------pdf转images------
    if pdf_path[-3:] == 'pdf':
        imgs_list = pdf2images(pdf_path)
    else:
        imgs_list = cv2.imread(pdf_path)

    # -----------处理图片开始----------
    if type(imgs_list) == numpy.ndarray:
        cv2.imwrite(crops_save_path + str(image_index) + '.png', imgs_list)
    else:
        for img in imgs_list:
            cv2.imwrite(crops_save_path + str(image_index) + '.png', img)
            image_index += 1

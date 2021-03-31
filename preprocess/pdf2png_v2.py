import fitz
import cv2
import os


def pdf2image(pdf_name):
    pdf = fitz.Document(pdf_name)
    imgs_list = []
    for i, pg in enumerate(range(0, pdf.pageCount)):
        page = pdf[pg]  # 获得每一页的对象
        trans = fitz.Matrix(2.0, 2.0)
        pm = page.getPixmap(matrix=trans, alpha=False)  # 获得每一页的流对象
        # pm.writePNG(dir_name + os.sep + base_name[:-4] + '_' + '{:0>3d}.png'.format(pg + 1))  # 保存图片
        img_path = pdf_name[:-4] + '_' + str(pg + 1) + '.png'
        pm.writePNG(img_path)  # 保存图片
        img = cv2.imread(img_path)
        imgs_list.append(img)
        os.remove(img_path)     # 删除图片
    pdf.close()
    return imgs_list


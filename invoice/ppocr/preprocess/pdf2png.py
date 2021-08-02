# coding=utf-8
from __future__ import print_function

import cv2
import fitz
import os


def pdf2png(pdf_op_path, pdf_name):
    # pdfDoc = fitz.open('pdf/in_4.pdf')
    pdfDoc = fitz.open(pdf_op_path)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        mat = fitz.Matrix(2, 2)  # zoom factor 2 in each direction
        rect = page.rect  # the page rectangle
        # mp = rect.tl + (rect.br - rect.tl) * 0.20  # its middle point
        mp = rect.tl + (rect.br - rect.tl) * 1  # its middle point
        clip = fitz.Rect(rect.tl, mp)  # the area we want
        pix = page.getPixmap(matrix=mat, clip=clip)

        # pix.writePNG('test' + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内
        pix.writePNG('imgs_of_pdf' + '/' + '%s.png' % pdf_name[:-4])  # 将图片写入指定的文件夹内

    return 'imgs_of_pdf' + '/' + '%s' % pdf_name[:-3] + 'png'


if __name__ == '__main__':
    # print(os.getcwd())
    pdfs_path = '../../pdf_dir'
    for pdf_name in os.listdir(pdfs_path):
        pdf_full_path = pdfs_path + "/" + pdf_name
        img_full_path = pdf2png(pdf_full_path, pdf_name)
        print('img_name: ', img_full_path)
        image = cv2.imread(img_full_path)
        # cv2.imshow('as', image)
        # cv2.waitKey()
        # print('file_name: ', file_name)
        # try:
        #     message, img = decode(image)
        #     print(message)
        # except Exception:
        #     print('识别失败')
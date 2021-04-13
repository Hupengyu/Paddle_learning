# coding=utf-8
from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import numpy as np
import cv2
import fitz
import os
import searcher_box


def format_data(data):
    res = {}
    list_1 = data.split(',')
    res['发票代码'] = list_1[2]
    res['发票号码'] = list_1[3]
    # 20190712 -> 2019年07月12日
    a_list = list(list_1[5])[:8]
    a_list.insert(4, '-')
    a_list.insert(7, '-')
    resp = ''.join(a_list)
    res['开票日期'] = resp
    res['税后价格'] = list_1[4]
    res['校验码'] = list_1[6]

    return res


def ser(close_image):
    # 寻找轮廓  opencv4.0和 3.0 findContours函数返回值发生变化
    # close2_image, contours, hierarchy = cv2.findContours(close_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv2.findContours(close_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 删选轮廓
    # 当前图片中面积最大的轮廓即为二维码
    max_area = 0
    max_index = 0
    for i in range(0, len(contours)):
        area = cv2.contourArea(contours[i])  # 外框面积  #外界矩阵的面积
        if area > max_area:
            max_index = i
            max_area = area
    if len(contours) == 0:
        return None

    qr_cnt = contours[max_index]
    return qr_cnt


# 递归填充黑色多边形
def fill_black_poly(poly_image):
    # 传入黑色多边形
    # 如果当前多边形找到的轮廓数量小于等于2，说明当前图形已经被完全填充，无需再进行填充
    # 否则说明当前多边形还存在空白区域，需要继续填充
    # poly_image, contours, hierarchy = cv2.findContours(poly_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(poly_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) <= 2:
        return poly_image
    else:
        minArea = -1;

        # 从轮廓中找到面积最小的轮廓进行填充
        for i in range(0, len(contours)):
            if cv2.contourArea(contours[i]) < minArea or minArea == -1:
                minArea = cv2.contourArea(contours[i])
                minIndex = i

        cv2.fillConvexPoly(poly_image, contours[minIndex], (0, 0, 0), 8, 0)

        # if self.trace_image:
        #     cv2.imwrite(self.trace_path + "008_fill_poly_" + str(minArea) + self.image_name, poly_image)

        return fill_black_poly(poly_image)


# 图片预处理，旋转方正并拉伸到407 * 407分辨率
def prepare_image(cut_image, qr_cnt):
    # 最小外接矩形
    min_area_rect = cv2.minAreaRect(qr_cnt)
    # 取角点
    box_points = cv2.boxPoints(min_area_rect)
    # 生成透视变换矩阵
    source_position = np.float32(
        [[box_points[1][0], box_points[1][1]], [box_points[2][0], box_points[2][1]],
         [box_points[0][0], box_points[0][1]], [box_points[3][0], box_points[3][1]]])
    target_position = np.float32([[0, 0], [407, 0], [0, 407], [407, 407]])
    transform = cv2.getPerspectiveTransform(source_position, target_position)
    # 进行透视变换
    transform_image = cv2.warpPerspective(cut_image, transform, (407, 407))

    return transform_image


def remove_other_line(image):
    # 循环写法
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j][0] < image[i, j][1] or image[i, j][0] < image[i, j][2]) and image[i, j][0] < 150:
                # 其他颜色值高于蓝色，且蓝色低于150，说明是其他颜色覆盖
                gray_image[i][j] = 255
            else:
                gray_image[i][j] = max(min(-1.5 * image[i, j][0] + 1.5 * image[i, j][1] + 1.5 * image[i, j][2], 255), 0)
    return gray_image


def get_color_qrcode(image):
    # 循环写法
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if (image[i, j][0] < image[i, j][1] or image[i, j][0] < image[i, j][2]) and image[i, j][0] < 150:
                print('image[i, j][0]')
                print(image[i, j][0])
                print('image[i, j][1]')
                print(image[i, j][1])
                print('image[i, j][2]')
                print(image[i, j][2])
                # 其他颜色值高于蓝色，且蓝色低于150，说明是其他颜色覆盖
                return "black"
            else:
                print('image[i, j][0]')
                print(image[i, j][0])
                print('image[i, j][1]')
                print(image[i, j][1])
                print('image[i, j][2]')
                print(image[i, j][2])
                return "blue"
    return 'unknow'


def improvement_remove_other_line(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ratio = 1.5
    b = np.ones((image.shape[0], image.shape[1]), np.int16)
    g = np.ones((image.shape[0], image.shape[1]), np.int16)
    r = np.ones((image.shape[0], image.shape[1]), np.int16)
    source_b, source_g, source_r = cv2.split(gray_image)

    b[:, :] = source_b[:, :]
    g[:, :] = source_g[:, :]
    r[:, :] = source_r[:, :]

    g_minus_b = g - b
    r_minus_b = r - b

    g_minus_b = np.where(g_minus_b > 0, 1000, 0)
    r_minus_b = np.where(r_minus_b > 0, 1000, 0)

    g_minus_b_150 = np.where((g_minus_b - b) >= 850, 255, 0)
    r_minus_b_150 = np.where((r_minus_b - b) >= 850, 255, 0)
    g_r_minus_b_150 = g_minus_b_150 + r_minus_b_150
    not_b_lt_150 = np.where(g_r_minus_b_150 >= 255, -9999, 0)

    b_plus_g_r = -1 * ratio * b + ratio * g + ratio * r
    b_plus_g_r = np.where(b_plus_g_r < 0, 0, b_plus_g_r)
    b_plus_g_r = np.where(b_plus_g_r > 255, 255, b_plus_g_r)
    b_result = not_b_lt_150 + b_plus_g_r
    b_result = np.where(b_result < -9000, 255, b_result)
    b_result = np.where(b_result < 0, 0, b_result)
    b_result = np.where(b_result > 255, 255, b_result)

    gray_image[:, :, 0] = b_result[:, :]
    gray_image[:, :, 1] = b_result[:, :]
    gray_image[:, :, 2] = b_result[:, :]

    return gray_image


def get_max_np(np):
    x_min = np[0][0]
    x_max = np[0][0]
    y_min = np[1][1]
    y_max = np[1][1]
    for i in np:
        x_min = i[0] if(i[0] < x_min) else x_min
        x_max = i[0] if(i[0] >= x_max) else x_max
        y_min = i[1] if(i[1] < y_min) else y_min
        y_max = i[1] if(i[1] >= y_max) else y_max

    return y_min, y_max, x_min, x_max

# ------------------------------------------
# 黑色二维码识别工具方法


def decode(img):
    contours, gray = searcher_box.prethreatment(img)
    rec = searcher_box.pick_rectangels(contours)

    message, img = searcher_box.decode_qrcodes(rec, gray)
    return message


def decode_original(im):
    # Find barcodes and QR codes
    decodedObjects = pyzbar.decode(im)

    # Print results
    for obj in decodedObjects:
        print('Type : ', obj.type)
        print('Data : ', obj.data, '\n')

    return decodedObjects, decodedObjects[0].data.decode("utf-8")


# Display barcode and QR code location
def display(im, decodedObjects):
    # Loop over all decoded objects
    for decodedObject in decodedObjects:
        points = decodedObject.polygon

        # If the points do not form a quad, find convex hull
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            hull = list(map(tuple, np.squeeze(hull)))
        else:
            hull = points;

        # Number of points in the convex hull
        n = len(hull)

        # Draw the convext hull
        for j in range(0, n):
            cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 1)

    # Display results
    # cv2.imshow("Results", im)
    # cv2.waitKey(0);


def pdf_to_png(img_path, file_name):
    # pdfDoc = fitz.open('pdf/in_4.pdf')
    pdfDoc = fitz.open(img_path)
    for pg in range(pdfDoc.pageCount):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        # zoom_x = 1.33333333  # (1.33333333-->1056x816)   (2-->1584x1224)
        # zoom_y = 1.33333333
        # zoom_x = 4  # (1.33333333-->1056x816)   (2-->1584x1224)
        # zoom_y = 4
        mat = fitz.Matrix(2, 2)  # zoom factor 2 in each direction
        rect = page.rect  # the page rectangle
        # mp = rect.tl + (rect.br - rect.tl) * 0.20  # its middle point
        mp = rect.tl + (rect.br - rect.tl) * 1  # its middle point
        clip = fitz.Rect(rect.tl, mp)  # the area we want
        pix = page.getPixmap(matrix=mat, clip=clip)

        # pix.writePNG('test' + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内
        pix.writePNG('test' + '/' + '%s.png' % file_name[:-4])  # 将图片写入指定的文件夹内

    return 'test' + '/' + '%s' % file_name[:-3] + 'png'
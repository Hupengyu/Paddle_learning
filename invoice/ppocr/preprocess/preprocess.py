import os

import numpy as np
import cv2


def show_img(img, win_name, ratio=0.5):
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    # cv2.destroyWindow(win_name)


# 分割图片
def cut_image(image, count=2):
    height, width = image.shape[0], image.shape[1]
    item_width = width
    item_height = int(height / count)
    crops_list = []
    # 裁剪坐标为[y0:y1, x0:x1]
    for i in range(0, count):
        box = (i * item_height, (i + 1) * item_height, 0, item_width)
        crop = image[box[0]:box[1], box[2]:box[3]]
        crops_list.append(crop)
    return crops_list


# 图像增强
def preprocess_image(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # show_img(gray, 'gray')

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((1, 1), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=5)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=5)  # 腐蚀
    # show_img(morphed, 'morphed')
    return morphed


# 过滤掉蓝色字体和红色印章
def img_blue_filter(image):
    # 红色印章的阈值也去掉
    lower_unblue1 = np.array([15, 0, 0])
    upper_unblue1 = np.array([80, 255, 255])

    lower_unblue2 = np.array([180, 0, 0])
    upper_unblue2 = np.array([255, 255, 255])

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_unblue1, upper_unblue1)  # 阈值内的设为255,其余为0
    mask2 = cv2.inRange(hsv, lower_unblue2, upper_unblue2)  # 阈值内的设为255,其余为0
    mask = cv2.bitwise_or(mask1, mask2)
    blue_mask = mask == 255  # 取mask中为255的设置为true

    unblue_areas = np.zeros(image.shape, np.uint8)  # 创建新画布
    unblue_areas[:, :] = (255, 255, 255)  # 画布喷白
    unblue_areas[blue_mask] = image[blue_mask]  # 将blue的像素点‘喷’到白色画布上
    # 返回Img中的蓝色像素点
    return unblue_areas


def detect_big_box(img):
    # print('进入检测big_box-----%s------', img)
    big_box = []
    # 读入图像
    # img = cv2.imread(img_path)
    # show_img(img, 'img')
    # img = img_blue_filter(img)

    # 计算图片面积
    img_h, img_w = img.shape[0:2]
    # img_area = img_w * img_h
    # print("img_area: ", img_area)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # edged_copy = edged.copy()
    # show_img(edged_copy, 'edged')

    kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=1)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=1)  # 腐蚀

    # 找轮廓
    morphed_copy = morphed.copy()
    cnts, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    else:
        print("Did not find contours\n")
        return

    # 对前几个大小的框进行验证
    for box in cnts:
        # 用周长的0.05倍作为阈值，对轮廓做近似处理，使其变成一个矩形
        # epsilon:指定的精度，也即是原始曲线与近似曲线之间的最大距离
        # epsilon = 0.001 * cv2.arcLength(box, True)
        # approx = cv2.approxPolyDP(box, epsilon, True)
        rect_bound = cv2.boundingRect(box)  # (84,233,1592,752)|(左上角(x,y),weight,height)
        # rect1 = cv2.minAreaRect(box)    # ((879.54,608.73), (1589.82,747.67),0.169)|(中心点(x,y),(weight,height),角度)
        # 坐标格式转换
        tuple_new = ([0, 0], [0, 0], 0)
        rect_res = list(tuple_new)
        rect_res[0][0] = rect_bound[0] + rect_bound[2] / 2
        rect_res[0][1] = rect_bound[1] + rect_bound[3] / 2
        rect_res[1][0] = rect_bound[2]
        rect_res[1][1] = rect_bound[3]
        rect_res[0] = tuple(rect_res[0])
        rect_res[1] = tuple(rect_res[1])
        rect_res[2] = float(0)
        tuple_rect = tuple(rect_res)
        box = np.int0(cv2.boxPoints(tuple_rect))
        # box = sorted(box, key=lambda x: (x[0], x[1]))  # 排序方式！！！优先级（a>b）

        # print('box: ', box)
        # 在原图的拷贝上画出轮廓
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
        # show_img(img_copy, img_path)
        # print('img_path: ', img_p0071Doc0002.pngath)
        # cv2.imwrite(img_path, img_copy)

        # 获取透视变换的原坐标
        if box.shape[0] is not 4:
            # print("Found a non-rect\n")
            continue
        src_coor = np.reshape(box, (4, 2))
        src_coor = np.float32(src_coor)

        # 右上,左上,左下,右下 坐标
        # (tr, tl, bl, br) = src_coor
        (lb, lt, rt, rb) = src_coor

        # print('src_coor: ', src_coor)

        # 计算宽
        w1 = np.sqrt((rt[0] - lt[0]) ** 2 + (rt[1] - lt[1]) ** 2)
        w2 = np.sqrt((rb[0] - lb[0]) ** 2 + (rb[1] - lb[1]) ** 2)
        # 求出比较大的w
        max_w = max(int(w1), int(w2))
        # 计算高
        h1 = np.sqrt((lb[0] - lt[0]) ** 2 + (lb[1] - lt[1]) ** 2)
        h2 = np.sqrt((rb[0] - rt[0]) ** 2 + (rb[1] - rt[1]) ** 2)
        # 求出比较大的h
        max_h = max(int(h1), int(h2))

        weight = max(max_h, max_w)  # 最大的是宽度
        height = min(max_h, max_w)  # 最小的是高度

        # area_box = max_h * max_w

        # length_ratio = weight / img_w  # outline_w：image_w 宽度比
        # area_ratio = area_box / img_area  # 面积比
        length_height_ratio = height / weight  # 高宽比

        # print('length_height_ratio: ', length_height_ratio)
        # print('length_ratio: ', length_ratio)
        # print('area_ratio: ', area_ratio)
        # -----------判断是否是发票计数框big_box------------
        # if 0.85 < length_ratio and 0.30 < area_ratio and 0.45 < length_height_ratio < 0.65:
        if 0.45 < length_height_ratio <= 0.55:  # 未框到二维码
            big_box = src_coor
            return big_box
        elif 0.55 < length_height_ratio < 0.65:  # 框到二维码
            src_coor[1][1] += 0.14 * height
            src_coor[2][1] += 0.14 * height
            big_box = src_coor
            # print('src_coor: ', src_coor)
            # img_copy = img.copy()
            # big_box = np.int64(big_box)
            # cv2.drawContours(img_copy, [big_box], -1, (255, 0, 0), 2)
            # show_img(img_copy, img_path)
            return big_box

    return big_box


def detect_big_box_test(img, img_path=''):
    # print('进入检测big_box-----%s------', img)
    big_box = []
    # 读入图像
    # img = cv2.imread(img_path)
    # show_img(img, 'img')
    # img = img_blue_filter(img)

    # 计算图片面积
    img_h, img_w = img.shape[0:2]
    img_area = img_w * img_h
    # print("img_area: ", img_area)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # edged_copy = edged.copy()
    # show_img(edged_copy, 'edged')

    kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=1)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=1)  # 腐蚀

    # 找轮廓
    morphed_copy = morphed.copy()
    cnts, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    else:
        print("Did not find contours\n")
        return

    # 对前几个大小的框进行验证
    for box in cnts:
        # 用周长的0.05倍作为阈值，对轮廓做近似处理，使其变成一个矩形
        # epsilon:指定的精度，也即是原始曲线与近似曲线之间的最大距离
        # epsilon = 0.001 * cv2.arcLength(box, True)
        # approx = cv2.approxPolyDP(box, epsilon, True)
        rect_bound = cv2.boundingRect(box)    # (84,233,1592,752)|(左上角(x,y),weight,height)
        # rect1 = cv2.minAreaRect(box)    # ((879.54,608.73), (1589.82,747.67),0.169)|(中心点(x,y),(weight,height),角度)
        # 坐标格式转换
        tuple_new = ([0, 0], [0, 0], 0)
        rect_res = list(tuple_new)
        rect_res[0][0] = rect_bound[0] + rect_bound[2] / 2
        rect_res[0][1] = rect_bound[1] + rect_bound[3] / 2
        rect_res[1][0] = rect_bound[2]
        rect_res[1][1] = rect_bound[3]
        rect_res[0] = tuple(rect_res[0])
        rect_res[1] = tuple(rect_res[1])
        rect_res[2] = float(0)
        tuple_rect = tuple(rect_res)
        box = np.int0(cv2.boxPoints(tuple_rect))
        # box = sorted(box, key=lambda x: (x[0], x[1]))  # 排序方式！！！优先级（a>b）

        # print('box: ', box)
        # 在原图的拷贝上画出轮廓
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
        # show_img(img_copy, img_path)
        # print('img_path: ', img_p0071Doc0002.pngath)
        # cv2.imwrite(img_path, img_copy)

        # 获取透视变换的原坐标
        if box.shape[0] is not 4:
            # print("Found a non-rect\n")
            continue
        src_coor = np.reshape(box, (4, 2))
        src_coor = np.float32(src_coor)

        # 右上,左上,左下,右下 坐标
        # (tr, tl, bl, br) = src_coor
        (lb, lt, rt, rb) = src_coor

        # print('src_coor: ', src_coor)

        # 计算宽
        w1 = np.sqrt((rt[0] - lt[0]) ** 2 + (rt[1] - lt[1]) ** 2)
        w2 = np.sqrt((rb[0] - lb[0]) ** 2 + (rb[1] - lb[1]) ** 2)
        # 求出比较大的w
        max_w = max(int(w1), int(w2))
        # 计算高
        h1 = np.sqrt((lb[0] - lt[0]) ** 2 + (lb[1] - lt[1]) ** 2)
        h2 = np.sqrt((rb[0] - rt[0]) ** 2 + (rb[1] - rt[1]) ** 2)
        # 求出比较大的h
        max_h = max(int(h1), int(h2))

        weight = max(max_h, max_w)  # 最大的是宽度
        height = min(max_h, max_w)  # 最小的是高度

        area_box = max_h * max_w

        length_ratio = weight / img_w  # outline_w：image_w 宽度比
        area_ratio = area_box / img_area  # 面积比
        length_height_ratio = height / weight   # 高宽比

        # print('length_height_ratio: ', length_height_ratio)
        # print('length_ratio: ', length_ratio)
        # print('area_ratio: ', area_ratio)
        # -----------判断是否是发票计数框big_box------------
        # if 0.85 < length_ratio and 0.30 < area_ratio and 0.45 < length_height_ratio < 0.65:
        if 0.45 < length_height_ratio <= 0.55:  # 未框到二维码
            big_box = src_coor
            return big_box
        elif 0.55 < length_height_ratio < 0.65:  # 框到二维码
            src_coor[1][1] += 0.14 * height
            src_coor[2][1] += 0.14 * height
            big_box = src_coor
            # print('src_coor: ', src_coor)
            # img_copy = img.copy()
            # big_box = np.int64(big_box)
            # cv2.drawContours(img_copy, [big_box], -1, (255, 0, 0), 2)
            # show_img(img_copy, img_path)
            return big_box

    return big_box


def detect_train_box(img_path):

    # 读入图像
    img = cv2.imread(img_path)

    # 计算图片面积
    # img_h, img_w = img.shape[0:2]
    # img_area = img_w * img_h

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    show_img(gray, 'gray', 0.2)
    # 高斯模糊，消除一些噪声
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # show_img(blur, 'blur', 0.2)

    # ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    # # 寻找边缘
    # edged = cv2.Canny(thresh, 128, 256)
    # # edged_copy = edged.copy()
    # show_img(edged, 'edged', 0.2)
    #
    # kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    # morphed = cv2.dilate(edged, kernel, iterations=1)  # 膨胀
    # morphed = cv2.erode(morphed, kernel, iterations=1)  # 腐蚀
    # show_img(morphed, 'morphed', 0.2)
    #
    # # 找轮廓
    # morphed_copy = morphed.copy()
    cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    else:
        print("Did not find contours\n")
        return

    # 对前几个大小的框进行验证
    for box in cnts:
        # 用周长的0.05倍作为阈值，对轮廓做近似处理，使其变成一个矩形
        # epsilon:指定的精度，也即是原始曲线与近似曲线之间的最大距离
        # epsilon = 0.001 * cv2.arcLength(box, True)
        # approx = cv2.approxPolyDP(box, epsilon, True)
        rect_bound = cv2.boundingRect(box)    # (84,233,1592,752)|(左上角(x,y),weight,height)
        # rect1 = cv2.minAreaRect(box)    # ((879.54,608.73), (1589.82,747.67),0.169)|(中心点(x,y),(weight,height),角度)
        # 坐标格式转换
        tuple_new = ([0, 0], [0, 0], 0)
        rect_res = list(tuple_new)
        rect_res[0][0] = rect_bound[0] + rect_bound[2] / 2
        rect_res[0][1] = rect_bound[1] + rect_bound[3] / 2
        rect_res[1][0] = rect_bound[2]
        rect_res[1][1] = rect_bound[3]
        rect_res[0] = tuple(rect_res[0])
        rect_res[1] = tuple(rect_res[1])
        rect_res[2] = float(0)
        tuple_rect = tuple(rect_res)
        box = np.int0(cv2.boxPoints(tuple_rect))
        # box = sorted(box, key=lambda x: (x[0], x[1]))  # 排序方式！！！优先级（a>b）

        # 在原图的拷贝上画出轮廓
        img_copy = img.copy()
        cv2.drawContours(img_copy, [box], -1, (255, 255, 0), 5)
        show_img(img_copy, 'drawContours', 0.2)

        # # 获取透视变换的原坐标
        # if box.shape[0] is not 4:
        #     # print("Found a non-rect\n")
        #     continue
        # src_coor = np.reshape(box, (4, 2))
        # src_coor = np.float32(src_coor)
        #
        # # 右上,左上,左下,右下 坐标
        # (tr, tl, bl, br) = src_coor
        #
        # # 计算宽
        # w1 = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
        # w2 = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
        # # 求出比较大的w
        # max_w = max(int(w1), int(w2))
        # # 计算高
        # h1 = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
        # h2 = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
        # # 求出比较大的h
        # max_h = max(int(h1), int(h2))
        #
        # weight = max(max_h, max_w)  # 最大的是宽度
        # height = min(max_h, max_w)  # 最小的是高度
        #
        # area_box = max_h * max_w
        #
        # length_ratio = weight / img_w  # 长度比
        # area_ratio = area_box / img_area  # 面积比
        # length_height_ratio = height / weight   # 长宽比
        #
        # # print(length_height_ratio)
        # # print(length_ratio)
        # # print(area_ratio)
        # # -----------判断是否是发票计数框big_box------------
        # if 0.85 < length_ratio < 0.98 and 0.30 < area_ratio < 0.8 and 0.45 < length_height_ratio < 0.50:
        #     big_box = src_coor
        #     return big_box


def invoices_num_det(img):
    # 先把图片的蓝色区域(主要是二维码)去除掉
    # img = filter_blue(img)

    src_coor_list = []

    invoice_num = 0

    # 计算图片面积
    img_h, img_w = img.shape[0:2]
    img_area = img_w * img_h

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # show_img(gray, 'gray')

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    kernel = np.ones((3, 3), np.uint8)  # 膨胀腐蚀的卷积核修改
    morphed = cv2.dilate(edged, kernel, iterations=2)  # 膨胀
    morphed = cv2.erode(morphed, kernel, iterations=2)  # 腐蚀
    # show_img(morphed, 'morphed')

    # 找轮廓
    # edged_copy = edged.copy()
    cnts, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    else:
        print("Did not find contours\n")
        return

    global box_tl_x_anchor
    box_tl_x_anchor = 500
    global box_tl_y_anchor
    box_tl_y_anchor = 0
    for box in cnts:
        # 用周长的0.05倍作为阈值，对轮廓做近似处理，使其变成一个矩形
        # epsilon:指定的精度，也即是原始曲线与近似曲线之间的最大距离
        # epsilon = 0.001 * cv2.arcLength(box, True)
        # approx = cv2.approxPolyDP(box, epsilon, True)
        rect = cv2.minAreaRect(box)
        box = np.int0(cv2.boxPoints(rect))

        # 在原图的拷贝上画出轮廓
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
        # show_img(img_copy, 'drawContours')

        # 获取透视变换的原坐标
        if box.shape[0] is not 4:
            # print("Found a non-rect\n")
            continue
        src_coor_int = np.reshape(box, (4, 2))  # 并不会对坐标进行排序
        src_coor = np.float32(src_coor_int)

        # 在原图的拷贝上画出轮廓
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [src_coor_int], -1, (255, 0, 0), 2)
        # show_img(img_copy, 'drawContours_src_coor')

        # 右上,左上,左下,右下 坐标
        (tr, tl, bl, br) = src_coor  # 你能保证坐标的顺序吗？

        four_peak_list = [bl, br, tl, tr]

        # x坐标排序
        peaks_sorted_x_list = sorted(four_peak_list, key=lambda x: x[0])
        min_x_tmp = peaks_sorted_x_list[0][0]
        max_x_tmp = peaks_sorted_x_list[3][0]

        # y坐标排序
        peaks_sorted_y_list = sorted(four_peak_list, key=lambda x: x[1])
        min_y_tmp = peaks_sorted_y_list[0][1]
        max_y_tmp = peaks_sorted_y_list[3][1]

        # ------------处理重复识别-------------
        if abs(min_y_tmp - box_tl_y_anchor) < 150 and abs(min_x_tmp - box_tl_x_anchor) < 150:
            continue

        # 计算宽
        max_w = max_x_tmp - min_x_tmp
        # 计算高
        max_h = max_y_tmp - min_y_tmp
        # 旋转的情况考虑
        max_w = max(max_w, max_h)
        max_h = min(max_w, max_h)
        # 计算面积
        area_box = max_h * max_w

        weight_ratio = max_w / img_w  # 宽度比
        area_ratio = area_box / img_area  # 面积比
        height_weight_ratio = max_h / max_w

        # print(weight_ratio)
        # print(area_ratio)
        # print(height_weight_ratio)
        # **************************************判断box_pre_elc***********************************
        if 0.85 < weight_ratio and 0.30 < area_ratio and 0.45 < height_weight_ratio < 0.65:
            src_coor_list_len = len(src_coor_list)  # box_pre_elc的个数
            if src_coor_list_len != 0:
                # **********************过滤掉相似的box_pre_elc，不相似的留着并相加*********************
                for i in range(0, src_coor_list_len):
                    dif = np.square(src_coor_list[i] - src_coor).sum(axis=1).sum(axis=0)  # 四个点横纵坐标的差值的平方和
                    # print('dif: ', dif)
                    dif_ratio = dif / area_box  # 偏离面积与当前面积的商
                    # print('dif_ratio: ', dif_ratio)    # 偏离率小于0.5的情况下，就是说他们相交的情况大于0.5,此时过滤掉！！！
                    if 0.0 <= dif_ratio < 0.5:
                        # print('---------------重复--------------')
                        continue
                    elif src_coor_list_len == i + 1:
                        invoice_num += 1
                        box_tl_x_anchor = min_x_tmp  # 如果box确实符合，就替换anchor
                        box_tl_y_anchor = min_y_tmp  # 如果box确实符合，就替换anchor
                    # print('判断结束')
                    # print('invoice_num: ', invoice_num)
                    # print('--------------不重复---------------')
            else:
                invoice_num = 1
                box_tl_x_anchor = min_x_tmp  # 初始化box_tl_x_anchor
                box_tl_y_anchor = min_y_tmp  # 初始化box_tl_y_anchor
            src_coor_list.append(src_coor)

    # if invoice_num < 1:
    #     invoice_num = 1

    return invoice_num


if __name__ == '__main__':
    pwd = os.getcwd()

    image_dir = os.path.join(os.getcwd(), "img_test")  # /home/pengyu/PycharmProjects/common_ocr_v2/img_test

    for img in os.listdir(image_dir):  # 遍历文件夹下每一个文件(xxx.pdf/xxx.jpg)
        img_abs_path = image_dir + '/' + img  # /home/pengyu/PycharmProjects/common_ocr_v2/img_test/1.jpg
        img = cv2.imread(img_abs_path)
        big_box = detect_big_box_test(img)
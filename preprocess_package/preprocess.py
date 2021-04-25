import numpy as np
import cv2
import os

pwd = os.getcwd()


def filter_blue(image):
    lower_unblue1 = np.array([0, 0, 0])
    upper_unblue1 = np.array([80, 255, 255])

    lower_unblue2 = np.array([124, 0, 0])
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


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


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
    # image_list = [image.crop(box) for box in box_list]
    return crops_list


# 保存分割后的图片
def save_images(image, crops_save_path, image_index):
    cv2.imwrite(crops_save_path + str(image_index) + '.png', image)


def detect_image_counts(img):
    # ------------处理重复识别-------------
    # 先把图片的蓝色区域(主要是二维码)去除掉
    img = filter_blue(img)

    src_coor_list = []

    invoice_num = 0

    # 计算图片面积
    img_h = img.shape[0]
    img_w = img.shape[1]
    img_area = img_w * img_h
    # if img_h > img_w:  # 旋转图片
    #     img = np.rot90(img)

    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 高斯模糊，消除一些噪声
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # show_img(gray, 'gray')

    # 寻找边缘
    edged = cv2.Canny(gray, 50, 120)
    # show_img(edged, 'edged')

    # 形态学变换，由于光照影响，有很多小的边缘需要进行腐蚀和膨胀处理
    # kernel = np.ones((1, 1), np.uint8)      # 膨胀腐蚀的卷积核修改
    # morphed = cv2.dilate(edged, kernel, iterations=5)   # 膨胀
    # morphed = cv2.erode(morphed, kernel, iterations=5)  # 腐蚀
    # show_img(morphed, 'morphed')

    # 找轮廓
    morphed_copy = edged.copy()
    cnts, _ = cv2.findContours(morphed_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 排序，并获取其中最大的轮廓
    if len(cnts) is not 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    else:
        print("Did not find contours\n")
        return

    global box_tl_x_anchor
    box_tl_x_anchor = 500
    global box_tl_y_anchor
    box_tl_y_anchor = 0
    for box in cnts:
        # print(box)
        # 用周长的0.05 倍作为阈值，对轮廓做近似处理，使其变成一个矩形
        # epsilon:指定的精度，也即是原始曲线与近似曲线之间的最大距离
        # 多边形可视化
        # epsilon = 0.001 * cv2.arcLength(box, True)
        # approx = cv2.approxPolyDP(box, epsilon, True)
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [approx], -1, (0, 0, 255), 3)
        # show_img(img_copy, 'approx')

        # 画矩形
        rect = cv2.minAreaRect(box)
        box = np.int0(cv2.boxPoints(rect))

        # 在原图的拷贝上画出轮廓
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
        # show_img(img_copy, 'drawContours')
        # img_copy = img.copy()
        # cv2.drawContours(img_copy, [box], -1, (0, 0, 255), 3)
        # show_img(img_copy, 'drawContours')
        # 获取透视变换的原坐标
        if box.shape[0] is not 4:
            print("Found a non-rect\n")
            continue
        src_coor = np.reshape(box, (4, 2))  # 并不会对坐标进行排序
        src_coor = np.float32(src_coor)

        # 右上,左上,左下,右下 坐标
        (tr, tl, bl, br) = src_coor   # 你能保证坐标的顺序吗？
        # min_x_tmp
        min_x_tmp = min(tl[0], tr[0])
        min_x_tmp = min(min_x_tmp, bl[0])
        min_x_tmp = min(min_x_tmp, br[0])
        # min_y_tmp
        min_y_tmp = min(tl[1], tr[1])
        min_y_tmp = min(min_y_tmp, bl[1])
        min_y_tmp = min(min_y_tmp, br[1])

        if abs(min_y_tmp - box_tl_y_anchor) < 150 and abs(min_x_tmp - box_tl_x_anchor) < 150:
            continue

        # 计算宽
        w1 = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
        w2 = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
        # 求出比较大的w
        max_w = max(int(w1), int(w2))
        # 计算高
        h1 = np.sqrt((bl[0] - tl[0]) ** 2 + (bl[1] - tl[1]) ** 2)
        h2 = np.sqrt((br[0] - tr[0]) ** 2 + (br[1] - tr[1]) ** 2)
        # 求出比较大的h
        max_h = max(int(h1), int(h2))

        max_length = max(max_h, max_w)  # 最大的是宽度
        min_length = min(max_h, max_w)  # 最小的是高度

        area_box = max_h * max_w

        length_ratio = max_length / img_w  # 长度比
        area_ratio = area_box / img_area  # 面积比
        length_height_ratio = min_length / max_length

        # print(length_ratio)
        # print(area_ratio)
        # print(length_height_ratio)
        # print('--------------ratio---------------')
        # **************************************判断box_pre_elc***********************************
        if 0.85 < length_ratio < 0.97 and 0.30 < area_ratio < 0.8 and 0.45 < length_height_ratio < 0.55:
            # print('--------------enter---------------')
            # print('area_box: ', area_box)
            # print("coor: ", src_coor)
            img_copy = img.copy()
            cv2.drawContours(img_copy, [box], -1, (255, 0, 0), 2)
            show_img(img_copy, 'drawContours')

            src_coor_list_len = len(src_coor_list)  # box_pre_elc的个数
            if src_coor_list_len != 0:
                # **********************过滤掉相似的box_pre_elc，不相似的留着并相加*********************
                for i in range(0, src_coor_list_len):
                    dif = np.square(src_coor_list[i] - src_coor).sum(axis=1).sum(axis=0)    # 四个点横纵坐标的差值的平方和
                    dif_ratio = dif / area_box  # 偏离面积与当前面积的商
                    if 0.0 <= dif_ratio < 0.5:  # 偏离率小于0.5的情况下，就是说他们相交的情况大于0.5,此时过滤掉！！！
                        # print('---------------重复--------------')
                        continue
                    elif src_coor_list_len == i + 1:    # 最后一个都不重合就相加
                        invoice_num += 1
                        box_tl_x_anchor = min_x_tmp     # 如果box确实符合，就替换anchor
                        box_tl_y_anchor = min_y_tmp     # 如果box确实符合，就替换anchor
                        print('invoice_num: ', invoice_num)
                        # print('--------------不重复---------------')
            else:   # 初始化第一个box_pre_elc
                invoice_num = 1
                box_tl_x_anchor = min_x_tmp  # 初始化box_tl_x_anchor
                box_tl_y_anchor = min_y_tmp  # 初始化box_tl_y_anchor
                print('invoice_num: ', invoice_num)
            src_coor_list.append(src_coor)

    # # 透视变换的目标坐标
    # dst_coor = np.array([[max_w - 1, 0], [0, 0], [0, max_h - 1], [max_w - 1, max_h - 1]], dtype=np.float32)
    #
    # # 求转换矩阵
    # trans_mat = cv2.getPerspectiveTransform(src_coor, dst_coor)
    # # 进行转换，将图中对应坐标的图片截取出来，并转换到dst_coor大小
    # warped = cv2.warpPerspective(img, trans_mat, (max_w, max_h))
    if invoice_num < 1:
        invoice_num = 1

    return invoice_num


if __name__ == '__main__':
    img_path = ''
    img = cv2.imread(img_path)

    detect_image_counts(img)

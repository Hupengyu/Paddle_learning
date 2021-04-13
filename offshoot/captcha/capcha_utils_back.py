import numpy as np
import cv2
import datetime
import os
import muggle_ocr
sdk = muggle_ocr.SDK(model_type=muggle_ocr.ModelType.OCR)


def predict_by_img(img):
    text = sdk.predict(image_bytes=img)
    print(text)
    return text


def filter(img):
    w, h = img.size
    for x in range(w):
        for y in range(h):
            # r, g, b, _ = img.getpixel((x, y))
            r, g, b= img.getpixel((x, y))
            if 190 <= r <= 255 and 170 <= g <= 255 and 0 <= b <= 140:
                img.putpixel((x, y), (0, 0, 0))
            if 0 <= r <= 90 and 210 <= g <= 255 and 0 <= b <= 90:
                img.putpixel((x, y), (0, 0, 0))
    img = img.convert('L').point([0] * 150 + [1] * (256 - 150), '1')
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    # path = "{}filter.jpg".format(nowTime)
    path = r"Z:\wxf\captcha\filter\{}filter.jpg".format(nowTime)
    img.save(path)
    return path


def getStandardDigit(img):
    '''
        返回标准的数字矩阵
    '''
    STD_WIDTH = 32  # 标准宽度
    STD_HEIGHT = 64

    height, width = img.shape

    # 判断是否存在长条的1
    new_width = int(width * STD_HEIGHT / height)
    if new_width > STD_WIDTH:
        new_width = STD_WIDTH
    # 以高度为准进行缩放
    resized_num = cv2.resize(img, (new_width, STD_HEIGHT), interpolation=cv2.INTER_NEAREST)
    # 新建画布
    canvas = np.zeros((STD_HEIGHT, STD_WIDTH))
    x = int((STD_WIDTH - new_width) / 2)
    canvas[:, x:x + new_width] = resized_num

    return canvas


def sort_by(elem, index):
    return elem[index]


def sort_by_X(elem):
    return elem[0]


def sort_by_Y(elem):
    return elem[1]


def sort_by_index(elem):
    return elem[4]


def cut_image(img):

    shape = img.shape
    hight = shape[0]
    wegiht = shape[1]
    h = int(hight / 3)
    cropped = img[0:h, 0:wegiht]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped3 = img[2 * h:3 * h, 0:wegiht]  # 裁剪坐标为[y0:y1, x0:x1]
    cropped2 = img[h:2 * h, 0:wegiht]  # 裁剪坐标为[y0:y1, x0:x1]
    text_len = -1
    target_image = cropped
    for i in [cropped, cropped2, cropped3]:
        bytes = cv2.imencode('.jpg', i)[1].tobytes()
        text = predict_by_img(bytes)
        if len(text) > text_len and '验证码' not in text:
            text_len = len(text)
            target_image = i

    target_shape = target_image.shape
    y0 = 0
    y1 = target_shape[0]
    x0 = int(target_shape[1] * 0.20)
    x1 = int(target_shape[1] - target_shape[1] * 0.10)
    target_image = target_image[y0:y1, x0:x1]
    # cv2.imshow("sd", target_image)
    # cv2.waitKey(0)
    return target_image


def context():
    pass


def get_coordinate(back_mask):

    # 反色 变为数字的掩模
    num_mask = cv2.bitwise_not(back_mask)
    # 中值滤波
    num_mask = cv2.medianBlur(num_mask, 1)

    # 寻找轮廓
    contours, hier = cv2.findContours(num_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 声明画布 拷贝自img
    canvas = cv2.cvtColor(num_mask, cv2.COLOR_GRAY2BGR)

    minWidth = 5  # 最小宽度
    minHeight = 15  # 最小高度

    edge_array = []
    # 检索满足条件的区域
    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        if w < minWidth or h < minHeight:
            # 如果不满足条件就过滤掉
            continue
        edge_array.append([x, y, w, h])

    # 的到的框坐标是乱序的，先根据X坐标排序
    edge_array.sort(key=sort_by_X)
    # 添加索引，与number中对应的数字
    for n, N in enumerate(edge_array):
        N.append(n)
    return edge_array, canvas


def process(path, number, source_img, name):
    img = cut_image(cv2.imread(path))
    lowerb = (0, 0, 116)
    upperb = (255, 255, 255)
    # 根据hsv阈值 进行二值化
    back_mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), lowerb, upperb)

    edge_array, canvas = get_coordinate(back_mask)

    print(len(edge_array))
    print(edge_array)
    # os.remove(path)
    # 框的数量少于number一律视为yellow
    if len(edge_array) < len(number):
        edge_array = yellow_number_process(source_img)

    for n, N in enumerate(edge_array):
        x, y, w, h, index = N
        cv2.rectangle(canvas, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=1)

    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    cv2.imwrite(r"Z:\wxf\captcha\box\res{}.jpg".format(nowTime), canvas)

    edge_array.sort(key=sort_by_Y)
    edge_array = edge_array[:4]
    edge_array.sort(key=sort_by_index)
    print(edge_array)

    flag = check_box(edge_array, number)
    if flag == "fail":
        return "fail"

    try:
        res = str(number[edge_array[0][4]]) + str(number[edge_array[1][4]]) + str(number[edge_array[2][4]]) + str(number[edge_array[3][4]])
    except:
        res = 'fail'
    return res

    # for n, N in enumerate(edge_array):
    #     x, y, w, h, index = N
    #     cv2.rectangle(canvas, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 255), thickness=1)
    # cv2.imshow("ss", canvas)
    # cv2.waitKey(0)


def yellow_number_process(source_img):
    low_yellow = np.array([0, 0, 167])
    high_yellow = np.array([255, 60, 255])

    hsv = cv2.cvtColor(np.asarray(source_img), cv2.COLOR_BGR2HSV)
    img = cv2.inRange(hsv, low_yellow, high_yellow)
    img = cut_image(img)
    # cv2.imshow("daa", img)
    # cv2.waitKey(0)
    edge_array, canvas = get_coordinate(img)

    return edge_array


def check_box(edge_array, number):
    if len(edge_array) != len(number):
        return "fail"

    max_w = max(edge_array[0][2], edge_array[1][2], edge_array[2][2], edge_array[3][2])
    min_w = min(edge_array[0][2], edge_array[1][2], edge_array[2][2], edge_array[3][2])
    if max_w - min_w > 8:
        return "fail"

    max_h = max(edge_array[0][3], edge_array[1][3], edge_array[2][3], edge_array[3][3])
    min_h = min(edge_array[0][3], edge_array[1][3], edge_array[2][3], edge_array[3][3])
    if max_h - min_h > 3:
        return "fail"
    # [[166, 0, 10, 25, 8], [224, 1, 15, 25, 11], [263, 0, 34, 26, 13], [358, 2, 14, 24, 17]]

    return "success"

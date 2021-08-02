import cv2
import numpy as np

from tools.infer.angle_config import AngleModelPb, AngleModelPbtxt

angleNet = cv2.dnn.readNetFromTensorflow(AngleModelPb, AngleModelPbtxt)  # dnn 文字方向检测


def angle_adjust(img):
    angle_rotate = angle_detect_dnn(img)
    img_rotated, angle = rotate(img, angle_rotate)
    return img_rotated, angle


def rotate(img, angle):
    # (h, w) = img.shape[:2]
    # center = (w // 2, h // 2)
    if angle == 90:
        img_rotated = np.rot90(img)
        # M = cv2.getRotationMatrix2D(center, 90, 1.0)
        # img_rotated = cv2.warpAffine(img, M, (w, h))
        # im = Image.fromarray(img).transpose(Image.ROTATE_90)
        # img = np.array(im)
    elif angle == 180:
        img_rotated = np.rot90(img, 2)

        # M = cv2.getRotationMatrix2D(center, 180, 1.0)
        # img_rotated = cv2.warpAffine(img, M, (w, h))
        # im = Image.fromarray(img).transpose(Image.ROTATE_180)
        # img = np.array(im)
    elif angle == 270:
        img_rotated = np.rot90(img, 3)
        # M = cv2.getRotationMatrix2D(center, 270, 1.0)
        # img_rotated = cv2.warpAffine(img, M, (w, h))
        # im = Image.fromarray(img).transpose(Image.ROTATE_270)
        # img = np.array(im)
    else:
        img_rotated = img
        angle = 0

    return img_rotated, angle


def angle_detect_dnn(img, adjust=True):
    """
    文字方向检测
    """
    h, w = img.shape[:2]
    ROTATE = [0, 90, 180, 270]
    if adjust:
        thesh = 0.05
        xmin, ymin, xmax, ymax = int(thesh * w), int(thesh * h), w - int(thesh * w), h - int(thesh * h)
        img = img[ymin:ymax, xmin:xmax]  ##剪切图片边缘

    inputBlob = cv2.dnn.blobFromImage(img,
                                      scalefactor=1.0,
                                      size=(224, 224),
                                      swapRB=True,
                                      mean=[103.939, 116.779, 123.68], crop=False);
    angleNet.setInput(inputBlob)
    pred = angleNet.forward()
    index = np.argmax(pred, axis=1)[0]
    return ROTATE[index]


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('imgs_tmp/001_270.jpg')
    img_rotated, angle = angle_adjust(img)
    show_img(img_rotated, 'img_rotated')
    print('angle: ', angle)

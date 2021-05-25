import cv2
import os


def show_img(img, win_name, ratio=0.3):
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def get_img_size(img):
    sp = img.shape
    print('common_size: ', (sp[0] * sp[1] * sp[2])/1024/1024/sp[2])


def os_size(img_path):
    img = cv2.imread(img_path)
    img_height, img_width = img.shape[0:2]
    os_size = os.path.getsize(img_path) / 1024 / 1024
    while os_size > 1:
        print('os_size: %s, %s ' % (img_height, img_width))
        img = cv2.resize(img, (int(img_width/2), int(img_height/2)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(img_path, img)
        show_img(img, 'img')


if __name__ == '__main__':
    img_path = 'file/001_270.jpg'
    os_size(img_path)

    # img = cv2.imread(img_path)
    # get_img_size(img)

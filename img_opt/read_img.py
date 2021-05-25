import cv2


def opt(img_path):
    img = cv2.imread(img_path)
    cv2.imwrite(img_path, img)


if __name__ == '__main__':
    opt('file/001_270.jpg')
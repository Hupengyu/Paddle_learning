import cv2


def show_img(img, win_name):
    img = cv2.resize(img, None, fx=0.3, fy=0.3)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
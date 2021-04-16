from preprocess import detect_image_counts, cut_image
import os
import cv2

pwd = os.getcwd()

image_index = 1


def cut_images_save(img, save_path):
    global image_index
    print('**********************开始处理图片*************************')
    wrap = detect_image_counts(img)
    print('wrap: ', wrap)
    if wrap == 2:
        print('-------------此图片有两张发票----------------')
        imgs_list = cut_image(img)
        for img in imgs_list:
            cv2.imwrite(save_path + str(image_index) + '.png', img)
            image_index += 1
    else:
        cv2.imwrite(save_path + str(image_index) + '.png', img)
        image_index += 1
    print('***************************************处理图片完成*************************************')


if __name__ == '__main__':
    file_path = './pictures/hsv_color_threshold.png'
    crops_save_path = './results/crops/'

    img = cv2.imread(file_path)
    cut_images_save(img, crops_save_path)


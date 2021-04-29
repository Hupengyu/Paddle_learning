from preprocess import detect_image_counts, cut_image
import os
import cv2

pwd = os.getcwd()

image_index = 1


def cut_images_save(img, if_show_pre=False, img_name='', save_path='./results/crops/'):
    global image_index
    print('**********************开始处理图片*************************')
    print('img_name: ', img_name)
    wrap = detect_image_counts(img, if_show_pre, img_name)
    print('wrap: ', wrap)
    if wrap == 2:
        print('-------------此图片有两张发票----------------')
        imgs_list = cut_image(img)
        for img in imgs_list:
            cv2.imwrite(save_path + str(image_index) + '.png', img)
            image_index += 1
    else:
        cv2.imwrite(save_path + str(image_index) + '.jpg', img)
        image_index += 1
    print('***************************************处理图片完成*************************************')


if __name__ == '__main__':
    file_path = './pictures/2021_4_29.jpg'
    crops_save_path = './results/crops/'

    img = cv2.imread(file_path)
    cut_images_save(img, False, 'ss', crops_save_path)


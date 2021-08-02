import cv2
from PIL import Image
import os


def show_img(img, win_name, ratio=0.3):
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)


def compress_img(image, img_path, cover_ori_img=True, img_size_thresh_KB=1000):
    os_size = os.path.getsize(img_path)  # 返回字节数
    img_size = os_size / 1024
    if img_size < img_size_thresh_KB:
        return image
    else:
        if cover_ori_img:
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_width, img_height = img.size
            while img_size > img_size_thresh_KB:
                img.thumbnail((int(img_width / 2), int(img_height / 2)), Image.ANTIALIAS)
                img.save(img_path)
                img = Image.open(img_path)
                img_width, img_height = img.size
                img_size = os.path.getsize(img_path) / 1024  # 返回字节数
            img_compressed = cv2.imread(img_path)
            return img_compressed
        else:
            # 将原图复制，使用copy的图片，并最后重新命名
            image = image.copy()
            img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_width, img_height = img.size
            while img_size > img_size_thresh_KB:
                img.thumbnail((int(img_width / 2), int(img_height / 2)), Image.ANTIALIAS)
                ori_img_name, img_suffix = os.path.splitext(img_path)  # ('imgs/01_oppo', '.jpg')
                copy_img_path = ori_img_name + '_copy' + img_suffix    # 修改名称
                img.save(copy_img_path)
                img = Image.open(copy_img_path)
                img_width, img_height = img.size
                img_size = os.path.getsize(copy_img_path) / 1024  # 返回字节数
            img_compressed = cv2.imread(copy_img_path)
            return img_compressed


if __name__ == '__main__':
    img_path = 'imgs/01_oppo.jpg'
    img = cv2.imread(img_path)
    img_compressed = compress_img(img, img_path)
    # show_img(show_img, 'show_img')
    # img = cv2.imread(img_path)
    # get_img_size(img)

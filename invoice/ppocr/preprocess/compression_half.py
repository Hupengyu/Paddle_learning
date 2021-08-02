import os
import cv2

infile_dir = os.path.join(os.getcwd(), "doc/pp/", "IMG_20191012_142414.jpg")
outfile_dir = os.path.join(os.getcwd(), "preprocess_result_imgs/")


class Compress:

    def __init__(self, infile_path, outfile_path):
        self.infile_path = infile_path
        self.img_name = infile_path.split('/')[-1]
        self.outfile_path = outfile_path

    def compress_cv(self, compress_rate=0.5, show=False):
        img = cv2.imread(self.infile_path)
        heigh, width = img.shape[:2]
        # 双三次插值
        img_resize = cv2.resize(img, (int(width * compress_rate), int(heigh * compress_rate)),
                                interpolation=cv2.INTER_AREA)
        cv2.imwrite(self.outfile_path + 'result_cv_' + self.img_name, img_resize)
        print("%s 已压缩，" % self.img_name, "压缩率：", compress_rate)
        if show:
            cv2.imshow(self.img_name, img_resize)
            cv2.waitKey(0)
        return


if __name__ == '__main__':
    compress = Compress(infile_dir, outfile_dir)

    # 使用opencv压缩图片
    compress.compress_cv()

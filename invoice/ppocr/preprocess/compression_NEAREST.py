import os
import PIL.Image as Image

infile_dir = os.path.join(os.getcwd(), "doc/pp/", "IMG_20191012_143744.jpg")
outfile_dir = os.path.join(os.getcwd(), "preprocess_result_imgs/")


class Compress:

    def __init__(self, infile_path, outfile_path):
        self.infile_path = infile_path
        self.img_name = infile_path.split('/')[-1]
        self.outfile_path = outfile_path

    def compress_pil(self, way=1, compress_rate=0.5, show=False):
        """
        img.resize() 方法可以缩小可以放大
        img.thumbnail() 方法只能缩小
        :param way:
        :param compress_rate:
        :param show:
        :return:
        """
        img = Image.open(self.infile_path)
        w, h = img.size
        # 方法一：使用resize改变图片分辨率，但是图片内容并不丢失，不是裁剪

        img_resize = img.resize((int(w*compress_rate), int(h*compress_rate)))
        resize_w, resieze_h = img_resize.size
        img_resize.save(self.outfile_path + 'result_' + self.img_name)
        print("%s 已压缩，" % self.img_name, "压缩率：", compress_rate)


if __name__ == '__main__':
    compress = Compress(infile_dir, outfile_dir)

    # 使用PIL压缩图片
    compress.compress_pil()

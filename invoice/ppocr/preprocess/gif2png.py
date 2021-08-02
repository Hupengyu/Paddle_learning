from PIL import Image


def iter_frames(im):
    try:
        i = 0
        while 1:
            im.seek(i)
            imframe = im.copy()
            if i == 0:
                palette = imframe.getpalette()
            else:
                imframe.putpalette(palette)
            yield imframe
            i += 1
    except EOFError:
        pass


def gif2png(image_path, url_path, invoice_file_name):
    im = Image.open(image_path)
    image_name = invoice_file_name[:-4] + '.png'
    image_path_now = url_path + '/' + image_name
    for i, frame in enumerate(iter_frames(im)):
        frame.save(image_path_now, **frame.info)
    return image_path_now


if __name__ == '__main__':
    image_path = '../imgs/code.gif'
    url_path = '../imgs'
    invoice_file_name = 'code.gif'
    invoice_file_name_suffix = invoice_file_name[-4]
    image_path_now = gif2png(image_path, url_path, invoice_file_name)
    print(image_path_now)

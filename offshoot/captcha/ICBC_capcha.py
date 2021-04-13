from capcha_utils import *
from io import BytesIO
from flask import Flask, request
from PIL import Image, ImageGrab

app = Flask(__name__)


@app.route('/captcha', methods=['POST'])
def application():
    file = request.files['file']
    number = request.form['number']
    img_bytes = file.read()
    stream = BytesIO(img_bytes)
    img = Image.open(stream)
    (w, h) = img.size
    print(w)
    if w != 365 and h != 66:
        img = img.resize((365, 66))
    return process(number, img)


if __name__ == '__main__':
    # img = ImageGrab.grab()
    print(type(im))
    img = Image.open(r'E:\pythonProjects\项目代码\工行支付\icbc_pay\wxf\captcha\image\20201026_11_10_36.jpg')
    (w, h) = img.size
    print(w)
    if w != 365 and h != 66:
        img = img.resize((365, 66))
    for i in range(4):
        res = process("6212860111698989", img)
        print(res)
    # app.run('0.0.0.0', port=8080)

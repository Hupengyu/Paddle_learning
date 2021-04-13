from capcha_utils_back import *
from io import BytesIO
from flask import Flask, request
from PIL import Image
app = Flask(__name__)


@app.route('/captcha', methods=['POST'])
def application():
    file = request.files['file']
    number = request.form['number']
    img_bytes = file.read()
    stream = BytesIO(img_bytes)
    img = Image.open(stream)
    nowTime = datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    img.save(r"Z:\wxf\captcha\image\{}filter.jpg".format(nowTime))
    path = filter(img)
    name = file.filename
    res = process(path, number, img, name)
    print(res)
    return res


if __name__ == '__main__':

    # app.run('0.0.0.0', port=8080)
    img = Image.open(r'E:\pythonProjects\项目代码\工行支付\icbc_pay\wxf\captcha\image\20201026_11_16_37.jpg')

    low_yellow = np.array([0, 0, 167])
    high_yellow = np.array([255, 60, 255])

    hsv = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2HSV)
    im = cv2.inRange(hsv, low_yellow, high_yellow)
    # img = cut_image(img)
    # cv2.imshow("daa", img)
    # cv2.waitKey(0)

    res = process(im, '6212860111698989', img, '')
    print(res)
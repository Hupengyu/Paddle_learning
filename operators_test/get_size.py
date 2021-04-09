import os

path = 'pictures/img.png'
img_size = os.path.getsize(path) / 1024 / 1024
print(img_size)
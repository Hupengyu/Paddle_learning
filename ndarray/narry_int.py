import numpy as np

box = np.zeros((2, 2))


print(box)
print(type(box))

box[0][0] = np.int0(0)
box[0][1] = np.int0(9.5)
box[1][0] = np.int0(5.6)
box[1][1] = np.int0(1.9)

box = box.astype(np.uint8)
print(box)
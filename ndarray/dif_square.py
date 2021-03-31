import numpy as np

# nd = np.array([[1655., 1044.], [65., 1037.], [68., 237.], [1658., 243.]])
# nd1 = np.array([[1650., 1049.], [69., 1037.], [68., 237.], [1670., 249.]])
# tmp = np.array([[1658., 1054.], [65., 1034.], [60., 238.], [1658., 245.]])
nd = np.array([[1, 1], [1, 1]])
nd1 = np.array([[2, 3], [4, 5]])
nd2 = np.array([[2, 3], [4, 5]])

nd_list = []
nd_list.append(nd1)
nd_list.append(nd2)


for i in range(0, len(nd_list)):
    dif = np.square(nd_list[i] - nd).sum(axis=1).sum(axis=0)
    print(dif)

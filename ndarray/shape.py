import numpy as np

# 手动填写一个
a = [1, 2, 3, 4, 5, 6, 7, 8]
a = np.array(a)
print(a)

b = a.reshape(-1, 2)
print(b)
# 随机生成一个
# b = np.random.randint(0, 10, (2, 3))  # 两行三列，元素从0到10

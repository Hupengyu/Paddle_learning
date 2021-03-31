import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)  # 从(-1,1)均匀取50个点
y = 2 * x

plt.plot(x, y)
plt.show()
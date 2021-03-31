import paddle
from paddle.vision.transforms import Compose, Resize, ColorJitter
from paddle.io import Dataset


BATCH_SIZE = 64
BATCH_NUM = 20

IMAGE_SIZE = (28, 28)
CLASS_NUM = 10


class MyDataset(Dataset):
    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples
        # 在 `__init__` 中定义数据增强方法，此处为调整图像大小
        self.transform = Compose([Resize(size=32)])

    def __getitem__(self, index):
        data = paddle.uniform(IMAGE_SIZE, dtype='float32')
        # 在 `__getitem__` 中对数据集使用数据增强方法
        # data = self.transform(data.numpy())

        label = paddle.randint(0, CLASS_NUM - 1, dtype='int64')

        return data, label

    def __len__(self):
        return self.num_samples


# 测试定义的数据集
custom_dataset = MyDataset(BATCH_SIZE * BATCH_NUM)

print('=============custom dataset=============')
for data, label in custom_dataset:
    print(data.shape, label.shape)
    break

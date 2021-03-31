import paddle
from paddle.vision.transforms import ToTensor
from paddle.io import Dataset, DataLoader

train_dataset = paddle.vision.datasets.MNIST(image_path='../dataset/train-images-idx3-ubyte.gz',
                                             label_path='../dataset/train-labels-idx1-ubyte.gz', mode='train',
                                             transform=ToTensor())
val_dataset = paddle.vision.datasets.MNIST(image_path='../dataset/t10k-images-idx3-ubyte.gz',
                                           label_path='../dataset/t10k-labels-idx1-ubyte.gz', mode='test',
                                           transform=ToTensor())

mnist = paddle.nn.Sequential(
    paddle.nn.Flatten(),
    paddle.nn.Linear(784, 512),
    paddle.nn.ReLU(),
    paddle.nn.Dropout(0.2),
    paddle.nn.Linear(512, 10)
)

# 预计模型结构生成模型对象，便于进行后续的配置、训练和验证
model = paddle.Model(mnist)

# 模型训练相关配置，准备损失计算方法，优化器和精度计算方法
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)

mnist.train()

epochs = 5

# 设置优化器
optim = paddle.optimizer.Adam(parameters=mnist.parameters())
# 设置损失函数
loss_fn = paddle.nn.CrossEntropyLoss()

for epoch in range(epochs):
    for batch_id, data in enumerate(train_loader()):
        x_data = data[0]  # 训练数据
        y_data = data[1]  # 训练数据标签
        predicts = mnist(x_data)  # 预测结果

        # 计算损失 等价于 prepare 中loss的设置
        loss = loss_fn(predicts, y_data)

        # 计算准确率 等价于 prepare 中metrics的设置
        acc = paddle.metric.accuracy(predicts, y_data)

        # 下面的反向传播、打印训练信息、更新参数、梯度清零都被封装到 Model.fit() 中

        # 反向传播
        loss.backward()

        if (batch_id + 1) % 900 == 0:
            print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id + 1, loss.numpy(),
                                                                            acc.numpy()))

        # 更新参数
        optim.step()

        # 梯度清零
        optim.clear_grad()


import paddle
from paddle.vision.transforms import ToTensor

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
model.prepare(paddle.optimizer.Adam(parameters=model.parameters()),
              paddle.nn.CrossEntropyLoss(),
              paddle.metric.Accuracy())

# 开始模型训练
model.fit(train_dataset,
          epochs=5,
          batch_size=64,
          verbose=1)

model.evaluate(val_dataset, verbose=0)

paddle.metric.accuracy()


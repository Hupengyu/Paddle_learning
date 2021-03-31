import paddle
import numpy as np

inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
linear = paddle.nn.Linear(10, 10)
inp = paddle.to_tensor(inp)
out = linear(inp)
loss = paddle.mean(out)
adam = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters())
out.backward()
adam.step()
adam.clear_grad()

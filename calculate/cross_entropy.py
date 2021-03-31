import paddle

input_data = paddle.rand(shape=[5, 5])
label_data = paddle.randint(0, 5, shape=[5,1], dtype="int64")
weight_data = paddle.rand([5])

loss = paddle.nn.functional.cross_entropy(input=input_data,
                                          label=label_data,
                                          weight=weight_data)
print(loss)
# [4.38418674]

str = []  # 有的题目要输出字符串，但是有时候list更好操作，于是可以最后list转string提交
for i in range(0, 5):
    str.append('M')
print('list_str: ', str)
print(type(str))
str1 = ''.join(str)
print('str: ', str1)
print(type(str1))

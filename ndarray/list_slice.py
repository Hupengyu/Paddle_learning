feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

a_vars = feature_names[-2:]     # 取后两个
b_vars = feature_names[:-2]     # 取到后两个之前

c_vars = feature_names[::]      # 正序取所有
d_vars = feature_names[::-1]    # 倒序取所有


print(a_vars)
print(b_vars)
print('----------------------------')
print(c_vars)
print(d_vars)

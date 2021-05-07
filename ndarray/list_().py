movie_name = ['加勒比海盗', '骇客帝国', '第一滴血', '指环王', '霍比特人', '速度与激情']
print('------删除之前--------')
for temp in movie_name:

    print(temp)
    del movie_name[2]
print('--------删除之后---------')
for temp in movie_name:
    print(temp)


# tup1 = ('physics', 'chemistry', 1997, 2000)
# tup2 = (1, 2, 3, 4, 5, 6, 7)
bound = (84, 233, 1592, 752)
# (x,y,weight,height)
minarea = ((879.5465698242188, 608.7394409179688), (1589.826171875, 747.67919921875), 0.16926325857639313)
# （中心点(x,y)，(weight,height), 角度）

# x = x + weight/2
# y = y + height/2
tuple_new = ([0, 0], [0, 0], 0)

rect = list(tuple_new)
rect_res = list(tuple_new)
# for i in range(2):
#     rect.append([])
#     for j in range(2):
#         rect[i].append(0)
print(rect)

rect[0][0] = bound[0] + bound[2]/2
rect[0][1] = bound[1] + bound[3]/2
rect[1][0] = bound[2]
rect[1][1] = bound[3]
rect[2] = float(0)

rect[0] = tuple(rect[0])
rect[1] = tuple(rect[1])

tuple_rect = tuple(rect)
# for index, rect_cell in enumerate(rect):
#     print('rect[]: ', rect_cell)
#     tuple_rect_cell = tuple(rect_cell)
#     rect_res[index] = tuple_rect_cell
# tuple_a = tuple(tuple([y for y in x]) for x in rect)
# for index, rect_cell in enumerate(rect):
#     print(type(rect_cell))
#     rect[index] = tuple(rect_cell)
#     print(rect[index])

print(rect)

print(tuple_rect)
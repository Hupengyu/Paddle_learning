bl = [64., 83.]

br = [1681., 184.]

tl = [5., 1021.]

tr = [1622., 1123.]

list_a = [bl, br, tl, tr]
print('list_a: ', list_a)

sorted_a = sorted(list_a, key=lambda x: x[0])
print('sorted_a: ', sorted_a[0][0])
print('sorted_a: ', sorted_a[3][0])

sorted_a = sorted(list_a, key=lambda x: x[1])
print('sorted_a: ', sorted_a[0][1])
print('sorted_a: ', sorted_a[3][1])

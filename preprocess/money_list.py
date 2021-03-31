res1_list = [['￥4473.79'], ['￥4608.00'], ['￥134.21']]

pri = {}

money_list = []
print('价格长度： ', len(res1_list))
print(res1_list)
for res_money in res1_list:
    res_money = ''.join(res_money)
    money_num = res_money.replace('￥', '')
    money_list.append(float(money_num))
print(money_list)
money_list.sort(reverse=True)
print(money_list)
# pri["税后价格"] = res1_list[1].replace('￥', '')
pri["税后价格"] = str(money_list[1])
print(pri["税后价格"])


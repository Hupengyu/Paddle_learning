import re

tax_ratio = ['13', '11%', '15.4%', '16%', '0.5%', '19%']
buyer_name = ['江苏鼎晟液压有限公司', '[2019]399号南京造币有限公司', '江苏鼎晟液压有限责任公司']
tax_num = ['123456789123456789', '12345678912456789', '123456789123M56K89']
names = ['*as', '*asasdas','*a6565s','asd']

for name in names:
    res1 = re.search('^(\*.*)', name)
    if res1 is not None:
        # print(type(res1))
        res = res1.group()
        print(res)
        continue

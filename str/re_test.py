import re

tax_ratio = ['13', '11%', '15.4%', '16%', '0.5%', '19%']
buyer_name = ['江苏鼎晟液压有限公司', '[2019]399号南京造币有限公司', '江苏鼎晟液压有限责任公司']
tax_num = ['123456789123456789', '12345678912456789', '123456789123M56K89']
names = ['*as', '*asasdas', '*a6565s', 'asd']
data = ['开系日期：2021年06月28日']


def re_test(str_data):
    for str_tmp in str_data:
        # res1 = re.search('^(\*.*)', name)
        res1 = re.search('[0-9]{1,4}年[0-9]{1,2}月[0-9]{1,2}日$', str_tmp)
        if res1 is not None:
            # print(type(res1))
            res = res1.group()
            print(res)
            continue


if __name__ == '__main__':
    re_test(data)

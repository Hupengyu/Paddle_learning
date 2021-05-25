import re

banks = ['开户行及账号：江苏银行徐州经济开发区支行60020188000023518', '工行徐州市开发区支行1106021409210088822', '发区支行110602140',
         '工行徐州市开发区支行1106021409210088822asdas', '开户行及账号60020188000023518']

banks1 = ['开户行及账号：yi60020188000023518']
banks2 = ['开户行及账号：江苏银行徐州经济开发区支行60020188000023518']

tmp = ['中国建设银行太仓分行32201997336050308284']



def print_banks(banks):
    for bank in banks:
        # bank_count_findall = re.findall('((?<=支行)\d{12,25})$', bank)
        bank_count_match = re.search('((?<=支行)\d{12,25})$', bank).group()
        print(bank_count_match)


def print_bank1s(banks):
    for bank in banks:
        # bank_count_findall = re.findall('((?<=支行)\d{12,25})$', bank)
        # purchaser_bank_num = re.search('^开户行及账号：.*\d{12,25}$', bank)
        purchaser_bank_num = re.search('.*\d{12,25}$', bank)
        # bank_count_match = re.search('^开户行及帐号：\d{12,25}$', bank).group()
        if purchaser_bank_num is not None:
            purchaser_bank_num = purchaser_bank_num.group().replace('开户行及账号：', '')
            print(purchaser_bank_num)


if __name__ == '__main__':
    # print_banks(banks)
    print_bank1s(tmp)

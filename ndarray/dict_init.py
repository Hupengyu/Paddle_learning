def invoice_res_init():
    res = {}
    res['InvoiceCode'] = '1'
    res['InvoiceNum'] = ''
    res['InvoiceDate'] = ''
    res['InvoiceType'] = ''
    res['TotalAmount'] = ''
    res['CheckCode'] = '1'
    return res


def dict_init():
    res = {}
    total_amount = {}
    total_tax = {}
    amount_in_figuers = {}
    # 初始化Key
    total_amount["TotalAmount"] = ''
    total_tax['TotalTax'] = ''
    amount_in_figuers['AmountInFiguers'] = ''
    res.update(total_amount)
    res.update(total_tax)
    res.update(amount_in_figuers)
    return res


if __name__ == '__main__':
    res_block = invoice_res_init()
    null_info_num = 0
    if res_block['InvoiceCode'] != '':
        null_info_num += 1
    if res_block['InvoiceNum'] != '':
        null_info_num += 1
    if res_block['InvoiceDate'] != '':
        null_info_num += 1
    if res_block['TotalAmount'] != '':
        null_info_num += 1
    if null_info_num > 2:
        print(null_info_num)
    # print(type(invoice_res_init()))
    # print(dict_init())
    # print(type(dict_init()))
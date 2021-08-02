"""
徐工发票信息定位版正则化
"""

import re
import numpy as np
from ppocr.postprocess.text.detector.link_boxes_invoice import BoxesConnector
from ppocr.utils.logging import get_logger

logger = get_logger()


class InvoiceBlock():
    """
    增值税发票结构化识别
    """

    def __init__(self, big_box, dt_boxes, rec_res, ori_img):
        self.rec_res = rec_res
        self.big_box = big_box  # big_box的坐标
        self.dt_boxes = dt_boxes  # 此时boxes已经按照先x,后y的顺序排完了
        self.ori_img = ori_img
        self.res = {}
        self.head_block = []
        self.purchaser_block = []
        self.commodity_name_block = []
        self.commodity_type_block = []
        self.commodity_unit_block = []
        self.commodity_num_block = []
        self.commodity_price_block = []
        self.commodity_amount_block = []
        self.commodity_taxrate_block = []
        self.commodity_tax_block = []
        self.seller_block = []

        self.invoice_type()
        self.total_amount()  # 合计金额, 合计税额, 价税合计(小写)
        self.check_code()  # 校验码
        # self.tax_ratio()  # 税率
        self.sorting_blocks()  # 分捡boxes
        self.invoice_head()
        self.purchaser()
        self.commodity()
        self.seller()
        # self.form_baidu()

    def total_amount(self):
        """
        TotalAmount（小写）, TotalTax, AmountInFiguers
        """
        # pri = {}
        # for i in range(self.N):
        #     txt = self.result[i][0].replace(' ', '')
        #     txt = txt.replace(' ', '')
        #     res1 = re.findall('￥[0-9]{1,8}.[0-9]{1,2}', txt)
        #     if len(res1) > 0:
        #         pri["税后价格"] = res1[0].replace('￥', '')
        #         self.res.update(pri)
        #         break
        total_amount = {}
        total_tax = {}
        amount_in_figuers = {}
        # 初始化Key
        total_amount["TotalAmount"] = ''
        total_tax['TotalTax'] = ''
        amount_in_figuers['AmountInFiguers'] = ''
        self.res.update(total_amount)
        self.res.update(total_tax)
        self.res.update(amount_in_figuers)

        res1_list = []
        for i in range(len(self.rec_res)):
            txt = self.rec_res[i][0].replace(' ', '')
            # txt = txt.replace(' ', '')
            txt = txt.replace('（', '(')
            txt = txt.replace('）', ')')
            txt = txt.replace(',', '.')
            txt = txt.replace('羊', '￥')
            txt = txt.replace('¥', '￥')
            res1 = re.findall('￥[0-9]{1,8}.[0-9]{1,2}', txt)
            if len(res1) == 0:
                res1 = re.findall('\(小写\)￥[0-9]{1,8}.[0-9]{1,2}', txt)
                if len(res1) == 0:
                    res1 = re.findall('\(小写\)[0-9]{1,8}.[0-9]{1,2}', txt)  # 考虑无法识别出￥的情况
            if len(res1) > 0:
                res1_list.append(res1[0])
                # print(res1_list)
                # print('价格长度： ', len(res1_list))
            else:
                continue

            if len(res1_list) > 1:  # 识别出前两个金额:TotalAmount（小写）, TotalTax
                money_list = []
                # print('价格长度： ', len(res1_list))
                # print(res1_list)
                for res_money in res1_list:
                    res_money = ''.join(res_money)
                    res_money = res_money.replace(',', '.')
                    # res_money = res_money.replace('(小写)￥', '')
                    money_num = res_money.replace('￥', '')
                    # money_num = res_money.replace('(小写)', '')
                    money_list.append(float('%.2f' % float(money_num)))
                    # money_list.append(float('{x:.2f}'.format(x=float(money_num))))
                money_list.sort()
                total_amount["TotalAmount"] = str(money_list[1])
                total_tax['TotalTax'] = str(money_list[0])
                amount_in_figuers['AmountInFiguers'] = str(round((money_list[0] + money_list[1]), 2))
                self.res.update(total_amount)
                self.res.update(total_tax)
                self.res.update(amount_in_figuers)
                break

    def check_code(self):
        """
        校验码识别
        """
        check = {}
        check['CheckCode'] = ''
        self.res.update(check)
        for i in range(len(self.rec_res)):
            txt = self.rec_res[i][0].replace(' ', '')
            # txt = txt.replace(' ', '')
            res = re.findall('校验码:[0-9]{20}', txt)
            res += re.findall('校验码[0-9]{20}', txt)
            res += re.findall('码:[0-9]{20}', txt)
            res += re.findall('码[0-9]{20}', txt)
            res += re.findall('[0-9]{20}', txt)
            if len(res) > 0:
                check['CheckCode'] = res[0].replace('校验码:', '').replace('校验码', '').replace('码:', '').replace('码', '').replace(
                    ':', '')
                self.res.update(check)
                break

    def sorting_blocks(self):
        """
        分捡信息，分块
        """
        big_box = self.big_box  # big_box的weight,height
        # anchor_coor = np.reshape(big_box, (4, 2))
        # anchor_coor = np.float32(anchor_coor)
        big_box = sorted(big_box, key=lambda x: (x[1], x[0]))  # 排序方式！！！优先级（a>b）
        big_box = np.reshape(big_box, (4, 2))
        big_box = np.float32(big_box)
        # 左上,右上,左下,右下 (4, 2)
        (lt, rt, lb, rb) = big_box
        anchor_weight = rb[0] - lt[0]
        anchor_height = rb[1] - lt[1]

        # -----------------------------------------
        # 满足purchaser_block的boxes的左上角坐标阈值
        lt_x_pur = lt[0] + anchor_weight * (60 / 1600)  # 750-160
        rt_x_pur = lt[0] + anchor_weight * (920 / 1600)  # max_x
        lt_y_pur = lt[1] - 50  # min_y - 50(印刷可能印到偏上的位置)
        rt_y_pur = lt[1] + anchor_height * (180 / 750)  # max_y

        # 满足seller_block的boxes的左上角坐标阈值
        lt_x_sell = lt[0] + anchor_weight * (50 / 1600)  # min_x
        rt_x_sell = lt[0] + anchor_weight * (800 / 1600)  # max_x
        lt_y_sell = lt[1] + anchor_height * (500 / 750)  # min_y    590
        rt_y_sell = lt[1] + anchor_height  # max_y

        # 满足commodity_block的boxes的左上角坐标阈值
        lt_x_com = lt[0]  # min_x
        rt_x_com = lt[0] + anchor_weight  # max_x
        lt_y_com = lt[1] + anchor_height * (200 / 750)  # min_y
        rt_y_com = lt[1] + anchor_height * (465 / 750)  # max_y
        # 满足commodity_name_block的boxes的左上角坐标阈值
        lt_x_com_name = lt_x_com  # min_x
        rt_x_com_name = lt_x_com + anchor_weight * (400 / 1600)  # max_x
        lt_y_com_name = lt_y_com  # min_y
        rt_y_com_name = rt_y_com  # max_y

        # 先从dt_boxes,res中把(boxes, txt)放在一起
        txts = [self.rec_res[i][0] for i in range(len(self.rec_res))]

        #     box[0][0], box[0][1], box[1][0], box[1][1],
        #     box[2][0], box[2][1], box[3][0], box[3][1]
        for idx, (box, txt) in enumerate(zip(self.dt_boxes, txts)):
            box_lt_x = box[0][0]  # 左上角x值
            box_lt_y = box[0][1]  # 左上角y值

            # 开始sort，把满足不同条件的boxes，带着txt放入不同的block中
            # box(ndarray) = [[ 630.   58.], [1078.   56.], [1078.  101.], [ 630.  104.]]
            if lt[0] < box_lt_x < rt[0] and box_lt_y < lt[1]:  # head_block
                self.head_block.append(txt)
                continue
            elif lt_x_pur < box_lt_x < rt_x_pur and lt_y_pur < box_lt_y < rt_y_pur:  # purchaser_block
                # purchaser情况需要考虑开户行的链接问题，所以处理方式不同:将满足条件的boxes放入一个ndarray,然后connect
                self.purchaser_block.append((box, txt))
                continue
            elif lt_x_sell < box_lt_x < rt_x_sell and lt_y_sell < box_lt_y < rt_y_sell:  # seller_block
                # purchaser情况需要考虑开户行的链接问题，所以处理方式不同:将满足条件的boxes放入一个ndarray,然后connect
                self.seller_block.append((box, txt))
                continue
            #     ***************commodity*****************
            elif lt_x_com_name < box_lt_x < rt_x_com_name and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_name_block
                self.commodity_name_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (400 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    600 / 1600) and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_type_block
                self.commodity_type_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (605 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    715 / 1600) and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_unit_block
                self.commodity_unit_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (715 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    875 / 1600) and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_num_block
                self.commodity_num_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (875 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    1035 / 1600) and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_price_block
                self.commodity_price_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (1035 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    1265 / 1600) and lt_y_com_name < box_lt_y < lt[1] + anchor_height * (
                    455 / 750):  # commodity_amount_block
                self.commodity_amount_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (1270 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    1355 / 1600) and lt_y_com_name < box_lt_y < rt_y_com_name:  # commodity_taxrate_block
                self.commodity_taxrate_block.append(txt)
                continue
            elif lt_x_com_name + anchor_weight * (1360 / 1600) < box_lt_x < lt_x_com + anchor_weight * (
                    1530 / 1600) and lt_y_com_name < box_lt_y < lt[1] + anchor_height * (
                    455 / 750):  # commodity_tax_block
                self.commodity_tax_block.append(txt)
                continue
            else:
                continue

    def invoice_head(self):
        """
        票头信息：发票代码，发票号码，开票日期
        """

        def code():
            """
            发票代码识别
            """
            No = {}
            No['InvoiceCode'] = ''
            self.res.update(No)
            for i in range(len(self.head_block)):
                txt = self.head_block[i].replace(' ', '')
                res1 = re.search('^(\d{10})$', txt)
                if res1 is None:
                    res1 = re.search('^(\d{12})$', txt)
                if res1 is None:
                    res1 = re.search('(?:(?<!\d)\d{10}(?!\d))', txt)
                if res1 is None:
                    res1 = re.search('(?:(?<!\d)\d{12}(?!\d))', txt)
                if res1 is not None:
                    No['InvoiceCode'] = res1.group()
                    self.res.update(No)
                    del self.head_block[i]
                    break
                else:
                    continue
            if No['InvoiceCode'] == '':
                for i in range(len(self.head_block)):
                    txt = self.head_block[i].replace(' ', '')
                    txt = txt.replace(' ', '')
                    res1 = re.findall('^N[0-9]{16,18}', txt)    # N123456780000000000
                    if len(res1) == 0:
                        res1 = re.findall('^No([0-9]{16,18})', txt)     # No123456780000000000
                    if len(res1) > 0:
                        res1 = res1[0][-10:]  # 得到前8位数字
                        No["InvoiceCode"] = res1
                        self.res.update(No)
                        del self.head_block[i]
                        break
                    else:
                        continue

        def number():
            """
            识别发票号码
            """
            nu = {}
            nu["InvoiceNum"] = ''
            self.res.update(nu)
            for i in range(len(self.head_block)):
                txt = self.head_block[i].replace(' ', '')
                txt = txt.replace(' ', '')
                res1 = re.search('^(\d{8})$', txt)
                # if res1 is None:
                #     res1 = re.search('(?:(?<!\d)\d{8}(?!\d))', txt)
                if res1 is not None:
                    nu["InvoiceNum"] = res1.group()
                    self.res.update(nu)
                    del self.head_block[i]
                    break
                else:
                    continue
            if nu["InvoiceNum"] == '':  # 如果直接找没有找到就使用N123456780000000000
                for i in range(len(self.head_block)):
                    txt = self.head_block[i].replace(' ', '')
                    txt = txt.replace(' ', '')
                    res1 = re.findall('^N[0-9]{16,18}', txt)    # N123456780000000000
                    if len(res1) == 0:
                        res1 = re.findall('^No([0-9]{16,18})', txt)     # No123456780000000000
                    if len(res1) > 0:
                        print('InvoiceNum:', res1)
                        res1 = res1[0].replace('N', '').replace('No', '')[:8]  # 得到前8位数字
                        nu["InvoiceNum"] = res1
                        self.res.update(nu)
                        del self.head_block[i]
                        break
                    else:
                        continue

        def date():
            """
            识别开票日期
            """
            da = {}
            da["InvoiceDate"] = ''
            self.res.update(da)
            for i in range(len(self.head_block)):
                txt = self.head_block[i].replace(' ', '')
                txt = txt.replace(' ', '')
                res1 = re.search('[0-9]{1,4}年[0-9]{1,2}月[0-9]{1,2}日$', txt)
                if res1 is not None:
                    da["InvoiceDate"] = res1.group()
                    self.res.update(da)
                    del self.head_block[i]
                    break
                else:
                    continue
            if da["InvoiceDate"] == '':
                for i in range(len(self.head_block)):
                    txt = self.head_block[i].replace(' ', '')
                    res1 = re.findall('[0-9]{1,4}年[0-9]{1,2}月[0-9]{1,2}日', txt)
                    if len(res1) > 0:
                        da["InvoiceDate"] = res1[0].replace('日期:', '')
                        self.res.update(da)
                        del self.head_block[i]
                        break
                    else:
                        continue

        code()  # 发票代码
        number()  # 发票号码
        date()  # 开票日期

    def purchaser(self):
        """
        购买方信息：名称，纳税人识别号，地址丶电话，开户行及账号
        """

        def handle_conn():
            img_height, img_width = self.ori_img.shape[0:2]
            connector = BoxesConnector(self.purchaser_block, img_width, img_height, max_dist=20,
                                       overlap_threshold=0.5)  # connect参数调整区域
            txts = connector.connect_boxes(if_txts=True)  # 合并同行boxes
            return txts

        def purchaser_name(purchaser_txts):
            """
            识别购方名称
            """
            Purchaser_Name = {}
            Purchaser_Name['PurchaserName'] = ''
            self.res.update(Purchaser_Name)
            for i in range(len(purchaser_txts)):
                txt = purchaser_txts[i].replace(' ', '')
                purchaser_name_txt = re.search('(.*有限公司|.*有限责任公司)$', txt)
                if purchaser_name_txt is not None:
                    purchaser_name = purchaser_name_txt.group().replace('名称：', '').replace('名称:', '').replace('称：', '').replace('称:', '').replace('（', '(').replace('）', ')')
                    Purchaser_Name['PurchaserName'] = purchaser_name
                    self.res.update(Purchaser_Name)
                    del purchaser_txts[i]
                    break
                else:
                    continue

        def purchaser_registerNum(purchaser_txts):
            """
            识别购方税号
            """
            Purchaser_RegisterNum = {}
            Purchaser_RegisterNum['PurchaserRegisterNum'] = ''
            self.res.update(Purchaser_RegisterNum)
            for i in range(len(purchaser_txts)):
                txt = purchaser_txts[i].replace(' ', '')
                purchaser_registerNum = re.search('(\d{5}[A-Z0-9]{10}|\d{8}[A-Z0-9]{10})$', txt)
                if purchaser_registerNum is None:
                    purchaser_registerNum = re.search('纳税人识别号：\d{5}[A-Z0-9]{10}|\d{8}[A-Z0-9]{10}', txt)
                if purchaser_registerNum is not None:  # 18=8*num+10num_or_alp*10
                    Purchaser_RegisterNum['PurchaserRegisterNum'] = purchaser_registerNum.group().replace('纳税人识别号：', '')
                    self.res.update(Purchaser_RegisterNum)
                    del purchaser_txts[i]
                    break
                else:
                    continue

        def purchaser_address(purchaser_txts):
            """
            识别购方地址及电话
            """
            Purchaser_Address = {}
            Purchaser_Address['PurchaserAddress'] = ''
            self.res.update(Purchaser_Address)
            for i in range(len(purchaser_txts)):
                txt = purchaser_txts[i].replace(' ', '')
                purchaser_address_txt = re.search('^地址、电话：.*', txt)
                if purchaser_address_txt is not None:
                    purchaser_address = purchaser_address_txt.group().replace('地址、电话：', '').replace('（', '(').replace(
                        '）', ')')
                    Purchaser_Address['PurchaserAddress'] = purchaser_address
                    self.res.update(Purchaser_Address)
                    del purchaser_txts[i]
                    break
                else:
                    continue

        def purchaser_bank_num(purchaser_txts):
            """
            识别购方账号
            """
            Purchaser_Bank_Num = {}
            Purchaser_Bank_Num['PurchaserBank'] = ''
            self.res.update(Purchaser_Bank_Num)
            for i in range(len(purchaser_txts)):
                txt = purchaser_txts[i].replace(' ', '')
                purchaser_bank_num = re.search('^开户行及账号：.*\d{12,25}$', txt)
                if purchaser_bank_num is None:
                    purchaser_bank_num = re.search('.*\d{12,25}$', txt)
                if purchaser_bank_num is not None:
                    purchaser_bank_num = purchaser_bank_num.group().replace('开户行及账号：', '').replace('（', '(').replace(
                        '）', ')')
                    Purchaser_Bank_Num['PurchaserBank'] = purchaser_bank_num
                    self.res.update(Purchaser_Bank_Num)
                    del purchaser_txts[i]
                    break
                else:
                    continue

        purchaser_txts = handle_conn()

        purchaser_name(purchaser_txts)
        purchaser_registerNum(purchaser_txts)
        purchaser_address(purchaser_txts)
        purchaser_bank_num(purchaser_txts)

    def seller(self):
        """
        销方信息：名称，纳税人识别号，地址丶电话，开户行及账号
        """

        def handle_conn():
            img_height, img_width = self.ori_img.shape[0:2]
            connector = BoxesConnector(self.seller_block, img_width, img_height, max_dist=50,
                                       overlap_threshold=0.5)  # connect参数调整区域
            txts = connector.connect_boxes(if_txts=True)  # 合并同行boxes,是否带回txt
            return txts

        def seller_name(seller_txts):
            """
            识别销方名称
            """
            Seller_name = {}
            Seller_name['SellerName'] = ''
            self.res.update(Seller_name)
            for i in range(len(seller_txts)):
                txt = seller_txts[i].replace(' ', '')
                # print('seller_txt: ', txt)
                seller_name = re.search('(.*有限公司|.*有限责任公司)$', txt)
                if seller_name is not None:
                    seller_name = seller_name.group().replace('名称：', '').replace('名称:', '').replace('称：', '').replace(
                        '称:', '').replace('（', '(').replace('）', ')')
                    Seller_name['SellerName'] = seller_name
                    self.res.update(Seller_name)
                    del seller_txts[i]
                    break
                else:
                    continue

        def seller_registerNum(seller_txts):
            """
            识别销方税号
            """
            Seller_RegisterNum = {}
            Seller_RegisterNum['SellerRegisterNum'] = ''
            self.res.update(Seller_RegisterNum)
            for i in range(len(seller_txts)):
                txt = seller_txts[i].replace(' ', '')
                seller_registerNum = re.search('^(\d{5}[A-Z0-9]{10}|\d{8}[A-Z0-9]{10})$', txt)
                if seller_registerNum is None:
                    seller_registerNum = re.search('纳税人识别号：\d{5}[A-Z0-9]{10}|\d{8}[A-Z0-9]{10}', txt)
                if seller_registerNum is not None:  # 18=8*num+10num_or_alp*10
                    # print('seller_registerNum.group(): ', seller_registerNum.group())
                    Seller_RegisterNum['SellerRegisterNum'] = seller_registerNum.group().replace('纳税人识别号：', '')
                    self.res.update(Seller_RegisterNum)
                    del seller_txts[i]
                    break
                else:
                    continue

        def seller_address(seller_txts):
            """
            识别销方地址及电话
            """
            Seller_Address = {}
            Seller_Address['SellerAddress'] = ''
            self.res.update(Seller_Address)
            for i in range(len(seller_txts)):
                txt = seller_txts[i].replace(' ', '')
                seller_address = re.search('^地址、电话：.*', txt)
                if seller_address is not None:
                    seller_address = seller_address.group().replace('地址、电话：', '').replace('（', '(').replace('）', ')')
                    Seller_Address['SellerAddress'] = seller_address
                    self.res.update(Seller_Address)
                    del seller_txts[i]
                    break
                else:
                    continue

        def seller_bank_num(seller_txts):
            """
            识别销方账号
            """
            Seller_Bank_Num = {}
            Seller_Bank_Num['SellerBank'] = ''
            self.res.update(Seller_Bank_Num)
            for i in range(len(seller_txts)):
                txt = seller_txts[i].replace(' ', '')
                seller_bank_num = re.search('^开户行及账号：.*\d{12,25}$', txt)
                if seller_bank_num is None:
                    seller_bank_num = re.search('.*\d{12,25}$', txt)
                if seller_bank_num is not None:
                    seller_bank_num = seller_bank_num.group().replace('开户行及账号：', '').replace('（', '(').replace('）', ')')
                    Seller_Bank_Num['SellerBank'] = seller_bank_num
                    self.res.update(Seller_Bank_Num)
                    del seller_txts[i]
                    break
                else:
                    continue

        seller_txts = handle_conn()

        seller_name(seller_txts)
        seller_registerNum(seller_txts)
        seller_address(seller_txts)
        seller_bank_num(seller_txts)

    def commodity(self):
        """
        货物信息：货物名称，规格型号，单位，数量，单价，金额，税率，税额
        """
        CommodityName = {}
        CommodityType = {}
        CommodityUnit = {}
        CommodityNum = {}
        CommodityPrice = {}
        CommodityAmount = {}
        CommodityTaxRate = {}
        CommodityTax = {}

        commodity_name = []
        commodity_type = []
        commodity_unit = []
        commodity_num = []
        commodity_price = []
        commodity_amount = []
        commodity_taxrate = []
        commodity_tax = []

        #  CommodityName
        name_len = len(self.commodity_name_block)
        if name_len == 0:
            pass
        else:
            name_exist = re.search('^(\*.*)', self.commodity_name_block[0])
            if name_exist is None:  # 如果货物清单是空的，那就返回name的值
                commodity_name.append(self.commodity_name_block[0].replace('（', '(').replace('）', ')'))
            else:
                j = 0
                while j < name_len:
                    name = re.search('^(\*.*)', self.commodity_name_block[j])  # 本次name
                    if name is None:
                        commodity_name[-1] = commodity_name[-1] + self.commodity_name_block[j].replace('（', '(').replace('）', ')')
                        j += 1
                        continue
                    else:
                        commodity_name.append(name.group().replace('（', '(').replace('）', ')'))
                        j += 1
                        continue

        for type in self.commodity_type_block:
            commodity_type.append(type)
        for unit in self.commodity_unit_block:
            commodity_unit.append(unit)
        for num in self.commodity_num_block:
            commodity_num.append(num)
        for price in self.commodity_price_block:
            commodity_price.append(price)
        for amount in self.commodity_amount_block:
            commodity_amount.append(amount)
        for taxrate in self.commodity_taxrate_block:
            commodity_taxrate.append(taxrate)
        for tax in self.commodity_tax_block:
            commodity_tax.append(tax)

        CommodityName['CommodityName'] = commodity_name
        CommodityType['CommodityType'] = commodity_type
        CommodityUnit['CommodityUnit'] = commodity_unit
        CommodityNum['CommodityNum'] = commodity_num
        CommodityPrice['CommodityPrice'] = commodity_price
        CommodityAmount['CommodityAmount'] = commodity_amount
        CommodityTaxRate['CommodityTaxRate'] = commodity_taxrate
        CommodityTax['CommodityTax'] = commodity_tax
        self.res.update(CommodityName)
        self.res.update(CommodityType)
        self.res.update(CommodityUnit)
        self.res.update(CommodityNum)
        self.res.update(CommodityPrice)
        self.res.update(CommodityAmount)
        self.res.update(CommodityTaxRate)
        self.res.update(CommodityTax)

    # def __call__(self):
    #     self.sorting_blocks()  # 分捡boxes
    #
    #     self.invoice_head()
    #     self.purchaser()
    # def form_baidu(self):
    #     dict = {}
    #     dict['words_result'] = self.res
    #     dict['log_id'] = uuid.uuid1()
    #     dict['words_result_num'] = len(self.res)
    #     self.res = dict
    def invoice_type(self):
        """
        发票类型：默认设置为空
        """
        invoice_type = {}
        invoice_type['InvoiceType'] = ''
        self.res.update(invoice_type)

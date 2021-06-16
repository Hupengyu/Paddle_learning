import re

txt = '爱克奇换热技术（太仓）有限公司'

res = re.search('(.*有限公司|.*有限责任公司)$', txt).group()

print(res)
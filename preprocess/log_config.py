import logging


def log_api():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y.%m.%d. %H:%M:%S',
                        # filename='parser_result.log',
                        # filemode='w'
                        )
    # logging模块由logger，handler，filter，fomatter四个部分组成

    # 获取一个logger对象
    logger = logging.getLogger(__name__)
    # 设置日志输出等级
    logger.setLevel(logging.INFO)
    # # 创建一个文件的handler
    # f_handler = logging.FileHandler("xxx.log")
    # f_handler.setLevel(logging.INFO)
    # 创建一个控制台的handler
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.INFO)
    # 设置日志的输出格式
    fmt = logging.Formatter("%(asctime)s-%(name)s-%(levelname)s-%(message)s")
    # 给handler绑定一个fomatter类
    # f_handler.setFormatter(fmt)
    c_handler.setFormatter(fmt)
    # 绑定一个handler
    # logger.addHandler(f_handler)
    logger.addHandler(c_handler)

    # 使用logger输出日志信息
    logger.info("info")


def log():
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(asctime)s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y.%m.%d %H:%M:%S',)
    # 获取一个logger对象
    logger = logging.getLogger(__name__)
    return logger

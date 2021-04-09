def fetcher(obj, index):
    return obj[index]


if __name__ == '__main__':
    x = 'spam'
    try:
        res = fetcher(x, 4)
        print(res)
    except Exception:
        print('got exception')

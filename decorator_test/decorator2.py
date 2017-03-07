#encoding=utf8
"""
硬构造一个debug_login()函数 调用形式稍微优美一些
但是这里print_debug和login属于内聚类型的耦合
"""

def login():
    print('in log')

def print_debug(func):
    def _decorator():
        print('enter the login')
        func()
        print('exit the login')
    return _decorator # function as return value

# debug_login = print_debug(login)

# debug_login()

print_debug(login)()

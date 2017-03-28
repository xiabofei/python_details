#encoding=utf8
"""
被装饰的函数带带参数的情况
"""
from ipdb import set_trace as st

def print_debug(func):
    def __decorator(ser):
        print('enter the login')
        func(ser)
        print('exit the login')
    return __decorator

@print_debug
def login(user):
    print('in login:' + user)

st()
login('jatsz')

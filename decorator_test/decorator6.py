#encoding=utf8

"""
装饰有返回值的函数
"""
import functools
from ipdb import set_trace as st

def print_debug(func):
    @functools.wraps(func)
    def __decorator(*args, **kwargs):
        print('enter the login')
        result = func(args[0])
        print('exit the login')
        return result
    return __decorator

@print_debug
def login(user):
    print('in login:' + user)
    msg = "success" if user == "jatsz" else "fail"
    return msg

st(context=21)
result1 = login('jatsz')
print result1

result2 = login('candy')
print result2

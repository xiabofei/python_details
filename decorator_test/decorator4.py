#encoding=utf8
"""
被装饰的函数带带参数的情况
"""

from ipdb import set_trace as st

def printdebug(func):
    def __decorator(user):
        print('enter the login')
        func(user)
        print('exit the login')
    return __decorator

@printdebug
def login(user):
    print('in login:' + user)

st()
login('jatsz')

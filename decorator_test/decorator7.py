#encoding=utf8

"""
多个装饰器 注意装饰器的顺序
"""
from ipdb import set_trace as st

def printdebug(func):
    def __decorator():
        print('enter the login')
        func()
        print('exit the login')
    return __decorator

def others(func):
    def __decorator():
        print 'in others'
        func()
        print 'out others'
    return __decorator

@others
@printdebug
def login():
    print('in login:')

@printdebug
@others
def logout():
    print('in logout:')

# st()
login()

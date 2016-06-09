#encoding=utf8

"""
装饰器本身有参数
"""
from ipdb import set_trace as st

def printdebug_level(level):
    def printdebug(func):
        def __decorator(user):
            print('enter the login, and debug level is: ' + str(level))
            func(user)
            print('exit the login')
        return __decorator
    return printdebug

@printdebug_level(level=5)
def login(user):
    print('in login:' + user)

st()
login('jatsz')

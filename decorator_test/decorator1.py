#encoding=utf8
"""
大前提 : 在python中函数也是一个对象,可以作为变量传递
熟悉函数名作为变量传递
"""
def login():
    print('in login')

def print_debug(func):
    print('enter the login')
    func()
    print('exit the login')

print_debug(login)

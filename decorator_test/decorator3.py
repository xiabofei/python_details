#encoding=utf8
"""
改进上一版的代码
引入语法糖
Syntax Sugar
decorator就是 : 使用函数作为参数并且返回函数的函数
在这里就对应  : 用login作为print_debug的参数
在使用上调用的还是login 可以理解为login被print_debug装饰
所以调用的主体还是login
"""

def print_debug(func):
    def __decorator():
        print('enter the login')
        func()
        print('exit the login')
    return __decorator

@print_debug
def login():
    print('in login')

login()
# print_debug(print_debug(login))() # 装饰了多层

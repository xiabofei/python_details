#encoding=utf8

"""
在函数中用yield
用斐波那契数列的例子
用set_trace跟踪一下
"""

from ipdb import set_trace as st

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        a ,b = b, a + b
        n = n + 1

st()
for n in fib(6):
    print n

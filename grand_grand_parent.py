#encoding=utf8

"""
从grand parent继承的private属性
python解释器是如何处理的
"""
from pprint import pprint

class A():
    def __init__(self):
        self.__private = 'private in A'

class B(A):
    def __init__(self):
        A.__init__(self)

class C(B):
    def __init__(self):
        B.__init__(self)
    def f(self):
        print self.__private

c = C()
pprint(dir(c))

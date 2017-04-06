#encoding=utf8
"""
https://laike9m.com/blog/li-jie-python-super,70/
讲解python中super的用法
"""
from ipdb import set_trace as st

class Root(object):
    def __init__(self):
        print "this is Root"


class B(Root):
    def __init__(self):
        print "enter B"
        super(B, self).__init__()
        print "leave B"


class C(Root):
    def __init__(self):
        print "enter C"
        super(C, self).__init__()
        print "leave C"


class D(B, C):
    pass


# st(context=21)
# d = D()
# print d.__class__.__mro__

class A(type):
    def __init__(cls):
        super(A,cls).__init__( (), {})

A()


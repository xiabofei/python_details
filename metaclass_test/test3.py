#encoding=utf8

from ipdb import set_trace as st

class A(type):
    def __new__(cls, a):
        print cls
    def __init__(cls, a):
        print cls

class B(object):
    def __new__(cls, b):
        print cls
    def __init__(cls, b):
        print cls

st(context=21)
A_type = A('a')
B_object = B('b')

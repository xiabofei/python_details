#encoding=utf8
"""
"""
from ipdb import set_trace as st

class UpperString(object):
    def __init__(self):
        self._value = ''
    def __get__(self, instance, klass):
        return self._value
    def __set__(self, instance, value):
        self._value = value.upper()

class MyClass(object):
    attribute = UpperString()

st(context=21)

instance_of = MyClass()
instance_of.attribute

instance_of.attribute = 'my value'
instance_of.attribute

instance_of.__dict__ = {}

#encoding=utf8
"""
http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python
Custom metaclass
创建real class for a metaclass
"""

from ipdb import set_trace as st

class UpperAttrMetaclass(type):
    """
    __new__ is the method called before __init__
    it's the method that creates the object and returns it
    while __init__ just initializes the object passed as parameter
    you rarely use __new__, except when you want to control how the object is created
    here the created object is the class, and we want to cutomize it
    so we override __new__
    you can do some stuff in __init__ too if you wish
    """
    def __new__(cls, clsname, bases, dct):
        uppercase_attr = {}
        for name, val in dct.items():
            if not name.startswith('__'):
                uppercase_attr[name.upper()] = val
            else:
                uppercase_attr[name] = val
        # return type.__new__(cls, clsname, bases, uppercase_attr)
        return super(UpperAttrMetaclass, cls).__new__(cls, clsname, bases, uppercase_attr)

__metaclass__ = UpperAttrMetaclass

st(context=21)
class Foo:
    bar = 'bip'

print hasattr(Foo, 'bar')
print hasattr(Foo, 'BAR')

f = Foo()
print f.BAR

#encoding=utf8

"""
可以用*args 和 **kwargs这种magic来修补
但是这是一种awful fix
因为任何参数都可以传进去了

总之这种继承多个parent的 怎么弄都是shit 尽量别这么搞
"""
from ipdb import set_trace as st

class BaseBase(object):
    def __init__(self, *args, **kwargs):
        print 'basebase'
        super(BaseBase, self).__init__()
    
class Base1(BaseBase):
    def __init__(self, *args, **kwargs):
        print 'base1'
        super(Base1, self).__init__(*args, **kwargs)

class Base2(BaseBase):
    def __init__(self, *args, **kwargs):
        print 'base2'
        super(Base2, self).__init__(*args, **kwargs)

class MyClass(Base1, Base2):
    def __init__(self, arg):
        print 'my base'
        super(MyClass, self).__init__(arg)

st(context=21)
m = MyClass(10)

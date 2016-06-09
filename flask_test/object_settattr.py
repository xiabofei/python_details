#encoding=utf8

"""
werkzeug/local.py
LocalProxy
为什么__init__中用的是object.__(self, '__name__', name)
而不是直接用self.__name__ = name这样的方式?
http://stackoverflow.com/questions/33041518/object-setattr-self-instead-of-setattrself
http://stackoverflow.com/questions/14756289/setattrobject-name-value-vs-object-setattr-name-value
self.x = x会调用class本身的__setattr__这样的hooks
如果该类的__setattr__方法被重写了 就会影响到self.x = x的执行
"""

class Test(object):
    def __init__(self, attr1, attr2):
        object.__setattr__(self, '__attr2__', attr2)
        self.__attr1__ = attr1

    def __setattr__(self, name, val):
        print "override the __setattr__ method and call self.__setattr__(self, %s, %s)" % (name, val)
        object.__setattr__(self, name, val)

from ipdb import set_trace as st
st()
t = Test('attr1','attr2')

print "t.__attr1__ : %s" % t.__attr1__
print "t.__attr2__ : %s" % t.__attr2__


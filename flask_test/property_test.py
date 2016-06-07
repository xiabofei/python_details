#encoding=utf8
"""
werkzeug/local.py
LocalStack
property
可以参考
https://docs.python.org/2/library/functions.html?highlight=property#property
property的两种用法
"""
class C(object):

    def __init__(self):
        self.__private = 'private'

    def __call__(self):
        print "__call__"

    def _get_private__(self):
        return self.__private

    def _set_private__(self, value):
        # self.__private = value
        object.__setattr__(self, self.__private, value)

    var = property(_get_private__, _set_private__)
    del _get_private__, _set_private__
    
    @property
    def other(self):
        """read only x's property"""
        return self.__private
    @other.setter
    def other(self, value):
        self.__private = value
    @other.deleter
    def other(self):
        del self.__private


c = C()
c()
# print c.var
# c.var = 1
# print c.var
# print c._get_private__()
print c.other
c.other = 1
print c.other

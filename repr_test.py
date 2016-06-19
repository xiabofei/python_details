#encoding=utf8

"""
werkzeug/routing/
Rule
__repr__方法
直接print某个对象时 打印出的信息
"""

class Test(object):
    def __repr__(self):
        return "<%s, called>" % (self.__class__.__name__)

t = Test()
print t

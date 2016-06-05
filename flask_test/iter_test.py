#encoding=utf8

"""
werkzeug/local.py里面Local类 试验其中的iter方法
给某个类构造一个__iter__方法
可以用for it in object这样的形式迭代
"""

class Local(object):

    def __init__(self):
        self.__storage__ = {}

    def __iter__(self):
        return iter(self.__storage__.keys())

l = Local()
l.__storage__[1] = 'a'
l.__storage__[2] = 'b'
l.__storage__[3] = 'c'

for it in l:
    print it

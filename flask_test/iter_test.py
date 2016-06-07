#encoding=utf8

"""
werkzeug/local.py
Local
def __iter__
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

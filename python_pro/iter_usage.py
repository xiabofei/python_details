# encoding=utf8
"""
迭代器iterators的用法
for ele in MyIterator(xxx)
上面这个场景下MyIterator(xxx)返回的内容由__iter__()决定
"""

from ipdb import set_trace as st

class MyIterator(object):
    def __init__(self, step):
        self.step = step
    def next(self):
        if self.step == 0:
            raise StopIteration
        self.step -= 1
        return self.step
    def __iter__(self):
        return self

st(context=21)
for ele in MyIterator(4):
    print ele

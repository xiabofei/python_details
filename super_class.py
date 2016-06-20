#encoding=utf8
"""
试验super class的初始化
"""

import pprint

class Parent():
    def __init__(self):
        self.__parent_private1 = 'pprivate1'
        self._parent_protect1 = 'pprotect1'
        self.parent_public1 = 'ppublic1'
    def __f_private():
        pass
    def _f_protect():
        pass
    def f_public():
        pass

class Child(Parent, object):
    def __init__(self):
        Parent.__init__(self)
        self.__child_private1 = 'cprivate1'

c = Child()
pprint.pprint(dir(c))

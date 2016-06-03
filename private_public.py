#encoding=utf8
class A(object):
    def __init__(self):
        self._A__private()
        self.__private()
        self.public()
    def __private(self):
        print 'A.__private()'
    def public(self):
        print 'A.public()'

class B(A):
    def __private(self):
        print 'B.__private'
    def public(self):
        print 'B.public()'

class C(A):
    def __init__(self):
        self.__private()
        self.public()
        self.__xbf = '__xbf inside C'
    def __private(self):
        print 'C.__private'
    def public(self):
        print 'C.public()'

def _single():
    print '_single()'

def single():
    print 'single()'

# b = B()
# c = C()
c = C()
print '\n'

__xbf__ = '__xbf__ outside'
print c.__xbf
print __xbf__

#encoding=utf8

from greenlet import greenlet
from greenlet import getcurrent as get_ident

def f1():
    print 12
    gr2.switch()
    print dir(get_ident())
    print "thread id in f1() " + str(get_ident())
    print 34
    gr2.switch()

def f2():
    print 56
    print "thread id in f2() "+str(get_ident())
    gr1.switch()
    print 78

gr1 = greenlet(f1)
gr2 = greenlet(f2)
gr1.switch()

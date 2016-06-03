#encoding=utf8

global1 = 'g1'
global2 = 'g2'

def f1():
    local1 = 'l1 in f1'
    local2 = 'l2 in f1'
    print locals()
    print globals()

def f2():
    local1 = 'l1 in f2'
    local2 = 'l2 in f2'
    print locals()
    print globals()

f1()
f2()

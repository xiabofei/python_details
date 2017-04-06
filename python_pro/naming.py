#encoding=utf8
"""
Expert Python Programming chapter4
本意是python中constant变量起名的问题

这里还引入一个binary bit-wise operations的技巧

简单场景:
    问题:
        查找某个constant是否在某堆constants中
    方法:
        const1 in [const1, const2, const3, const4, ... ] 

复杂场景:
    问题:
        查找某几个constant组合是否在某堆constants中
    传统方法:
        const1 in constants or const2 in constants
    binary bit-wise operations方法:
        用下面这种方法技巧的关键点在于
            1) 用dict数据结构给每个constant对应一个integer 这个integer只有某一个bit是1
            2) 为了保证distinct integer, 用 << operator 结合len(OPTIONS) 这样肯定不会重复

    
    

"""

OPTIONS = {}

from ipdb import set_trace as st
st(context=21)

def register_option(name):
    return OPTIONS.setdefault(name, 1 << len(OPTIONS))

def has_option(options, name):
    return bool(options & name)

# now defining options
BLUE = register_option('BLUE')
RED = register_option('RED')
WHITE = register_option('WHITE')

# let's try them
SET = BLUE | RED
print has_option(SET, BLUE)
print has_option(SET, WHITE)

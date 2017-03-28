#encoding=utf8
"""
yield可以将一个复杂的功能 拆分成多个简单的处理环节
比如下面这个例子 adder和power 是拆分出来的两个子功能

想象一下如果有如下的需求:
    要对一个list的values分别进行adder和power的处理, 如果用传统的方法
    for i in range(len(values)):
        val = values[i]
        tmp = adder(val)
        values[i] = power(tmp)
    上面这种实现虽然也分开了adder和power, 但是如果要去掉其中一个环节就有些复杂, 耦合度比较高
    res = adder(power(values))
    返回一个迭代器, 首先耦合度降低了, 其次可以将res这个迭代器(而不是list)作为参数传递给其他函数

通过这个例子体会书上的话:
    "generators should be considered every time you deal with a function
    that returns a sequence or works in a loop"
    如果返回的是一个list, 而且list中的每个元素都需要经过多个复杂的处理, 则可以用generator的语法
"""

def power(values):
    for value in values:
        print 'powering %s' % value
        yield value

def adder(values):
    for value in values:
        print 'adding to %s' % value
        if value % 2 == 0:
            yield value + 3
        else:
            yield value + 2

elements = [1, 4, 7, 9, 12, 19]

res = adder(power(elements))

print [ e for e in res ]

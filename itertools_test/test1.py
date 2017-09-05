#encoding=utf8

from itertools import chain

def generator1():
    for item in 'abcdef':
        yield item

def generator2():
    for item in '123456':
        yield item

generator3 = chain(generator1(), generator2())
for item in generator3:
    print item

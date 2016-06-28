#encoding=utf8
"""
学习memoryview的用法
http://stackoverflow.com/questions/18655648/what-exactly-is-the-point-of-memoryview-in-python
"""
from time import time
from ipdb import set_trace as st
# copy
for n in (100000, 200000, 300000, 400000):
    data = 'x'*n
    start = time()
    while data:
        data = data[1:]
    print 'bytes', n, time()-start
# memoryview
for n in (100000, 200000, 300000, 400000):
    data = 'x'*n
    start = time()
    b = memoryview(data)
    while b:
        b = b[1:]
    print 'memoryview', n, time()-start

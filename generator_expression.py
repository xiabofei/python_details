#encoding=utf8

import time

lst = range(10000)

# c1 = 0
t1 = time.time()
it1 = (x%2 for x in lst if x%2==0)
try:
    while True:
        val = next(it1)
        # c1 += 1
except StopIteration:
    pass
print time.time()-t1
# print c1

# c2= 0
t2 = time.time()
it2 = (x for x in (y%2 for y in lst) if x==0)
try:
    while True:
        val = next(it2)
        # c2 += 1
except StopIteration:
    pass
print time.time()-t2
# print c2

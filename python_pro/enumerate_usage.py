# encoding=utf8
"""
enumerate的应用场景
解决for xxx in list这种场景下还需要同时迭代index的场景
"""

seq = ["one", "two", "three"]
# 1. 正确的用法
for i,ele in enumerate(seq):
    seq[i] = '%d: %s' % (i, seq[i])
print seq

seq = ["one", "two", "three"]
# 2. list comprehensions 和 enumerate的结合用法
def _treatment(pos, element):
    return '%d: %s' % (pos, element)
seq = [ _treatment(i, ele) for i,ele in enumerate(seq) ] 
print seq

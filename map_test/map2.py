# encoding=utf8
# 学习map函数的用法
# https://my.oschina.net/zyzzy/blog/115096
# map(f,iter)的"并行"效果
# 下面的例子并行处理三个list
# l3的长度比l1和l2长, 则l1和l2对应位置补齐为None
# 这个函数应该慎重使用 否则当并行处理元素的长度不确定的时候 结果不可知

from ipdb import set_trace as st

def abc(a, b, c):
    return a*10000 + b*100 + c

l1 = [11,22,33]
l2 = [44,55,66]
l3 = [77,88,99,10]

st()
print map(abc, l1, l2, l3)

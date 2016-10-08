# encoding=utf8
# 学习map函数的用法
# https://my.oschina.net/zyzzy/blog/115096
# map(f,iter)的最简单用法
# 用f对iter元素执行迭代操作

def add100(x):
    return x+100

dat_list = [1,2,3]
print map(add100, dat_list)

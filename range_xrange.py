#encoding=utf8
"""
测试range和xrange对内存占用的情况
用memory_profiler方法
"""
@profile
def test():
    n = 10000
    for x in range(n): pass
    for x in xrange(n): pass

test()

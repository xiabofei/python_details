# encoding=utf8
# python的lambda表达式相关
# 资料来自:
# https://www.zhihu.com/question/20125256
# lambda表达式可以简单理解为"匿名函数"
# 通过几个具体的例子去学习

# 1. lambda表达式对list元素迭代操作
print map(lambda x:x+1, [y for y in range(10)])
        
# 2. lambda表达式对多维数组排序
s = [('a',3),('b',2),('d',4),('c',1)]
print sorted(s, key=lambda x:x[1])

# 3. lambda表达式与filter结合
l = ['foo','bar','far']
print filter(lambda x:'f' in x, l)

# 4. lambda表达式与reduce结合:迭代处理
# from ipdb import set_trace as st
# st()
print reduce(lambda a, b: a * b, xrange(1, 5))

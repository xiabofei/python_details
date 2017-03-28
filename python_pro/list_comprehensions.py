# encoding=utf8
"""
比较列表推导的效率
如果是append这种 影响效率的一个因素是 在每个loop内需要判断sequence的哪个部分需要变化
如果用list comprehension则不用去维护
"""
NUM = 50000000

# 1. 低效写法
evens = []
i = 0
while i<NUM:
    if i%2 == 0:
        evens.append(i)
    i += 1

# 2. 高效写法
# evens = [ i for i in range(NUM) if i%2==0 ]

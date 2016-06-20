#encoding=utf8
"""
试验一下对象的"静态变量"
"""
class Test(object):
    var1 = 1
    def __init__(self):
        self.var2 = 2

print Test.var1
print Test().var2
print Test.var2

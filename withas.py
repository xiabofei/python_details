#encoding=utf8
"""
测试with as在python中的用法 可以选择性的处理异常
"""
import sys

class test:
    def __enter__(self):
        print("enter")
        return 1
    def __exit__(self, type, value, traceback):
        print("exit")
        return isinstance(value, NameError)

with test() as t:
    print("t is not the result of test()")
    print("t is 1 ".format(t))
    raise NameError("hi NameError")
    # raise TypeError("hi TypeError")

sys.exit()

#encoding=utf8

"""
werkzeug/local.py
Local
__slots__
查阅资料:
http://www.liaoxuefeng.com/wiki/001374738125095c955c1e6d8bb493182103fac9270762a000/0013868200605560b1bd3c660bf494282ede59fee17e781000
可以尝试把__slots__这一行屏蔽掉 看运行结果不同
"""

class Student(object):
    __slots__ = ('name', 'set_name', 'set_other')
    pass

# 实例动态绑定变量
s = Student()
s.name = 'xbf'
print s.name

# 实例动态绑定方法
def set_name(self, name):
    self.name = name
from types import MethodType
s.set_name = MethodType(set_name, s, Student)
s.set_name('xxx')
print s.name

# 给一个实例绑定的方法 另一个实例是无法看到的
try:
    s2 = Student()
    s2.name = 1
    print "s2.name : " + str(s2.name)
    s2.set_name('yyy')
except Exception as e:
    print(e)

# 给class绑定方法
Student.set_name = MethodType(set_name, None, Student)
s2.set_name('yyy')
print s2.name


# 类在继承时__slots__ 不会被继承下去
class Student2(Student):
    pass

s2 = Student2()
s2.newattr = 2
print s2.newattr


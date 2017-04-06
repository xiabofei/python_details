#encoding=utf8
"""
1)
@classmethod修饰的函数 隐式传入的第一个参数是classobj 
在subclass中method传入的第一个参数是subclassobj
2)
@staticmethod修饰的函数 没有隐式传入的函数
3)
什么修饰都没有就是instancemethod 隐式传入的第一个参数是instanceobj 
在subclass中method传入的第一个参数是subclass生成的instanceobj
4)
如果在subclass中重写这几种method会怎样
重写同名的方法就是被覆盖了
"""

class P(object):

    @classmethod
    def f1(cls):
        print cls 

    def f2(self):
        print self

    @staticmethod
    def f3(para):
        print para

class C(P):
    @staticmethod
    def f3(para1, para2):
        print para1,para2



P.f1()
P().f2()
P.f3('parent')
print ''
C.f1()
C().f2()
C().f3('child','child2')

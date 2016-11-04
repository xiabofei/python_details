#encoding=utf8
# 看cch的代码中modeling/similarity.py中用到下面略不熟悉的用法
# sklearn.tree.DecisionTreeClassifier.__init__.__func__
# 结论 : 这种用法的目的是检验给某个class初始化的参数是否正确
# 理解 : 这里__init__只是一个配角 其可以是class中定义的任何一个method
#        这里主要理清几种调用形式:
#        情形1 : "某个类实例.某个类方法" 这种调用, 会自动将该方法的第一个参数默认为'self'
#        情形2 : "某个类实例.某个类方法.__func__"这种调用, 会将这个方法与类实例解除绑定 第一个参数想传什么传什么
#        再回到cch代码的情景 目的是想检验:
#            喂给sklearn.tree.DecisionTreeClassifier初始化的参数是否正确
#            而初始化的参数都在函数__init__里面
#            因此'xxx.__init__.__func__'就可以检验类初始化的参数


import sys, traceback
import inspect

class C1():
    def __init__(self, arg1='a', arg2='b', arg3='c'):
        print "    self: " + str(self)
        print "    arg1: " + str(arg1)
        print "    arg2: " + str(arg2)
        print "    arg3: " + str(arg3)

def validate_para(f):
    print "    " + str(f)
    arg_spec = inspect.getargspec(f)
    print "    " + str(arg_spec.args)

# 情景1
print "C1().__init__(1):"
C1().__init__(1)

# 错误情况1
print "C1.__init__(1):"
try:
    C1.__init__(1)
except:
    print "    " + str(traceback.print_exc())

# 情景2
print "C1().__init__.__func__(1):"
C1().__init__.__func__(1)

# 验证参数
print "validate类初始化需要的参数信息:"
validate_para(C1.__init__.__func__)

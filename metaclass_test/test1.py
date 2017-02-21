#encoding=utf8
"""
http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python
Custom metaclasses的部分
体会module中'__metaclass__'用法
"""
# from ipdb import set_trace as st

def upper_attr(future_class_name, future_class_parents, future_class_attr):
    """
    Return a class object, with the list of its attribute turned into uppercase
    """
    # st(context=21)
    uppercase_attr = {}
    for name, val in future_class_attr.items():
        # 手动把module中定义的类的属性大小写给修改了
        if not name.startswith('__'):
            uppercase_attr[name.upper()] = val
        else:
            uppercase_attr[name] = val
    return type(future_class_name, future_class_parents, uppercase_attr)

__metaclass__ = upper_attr

class Foo():
    bar = 'bip'

print hasattr(Foo, 'bar')
print hasattr(Foo, 'BAR')

f = Foo()
print f.BAR

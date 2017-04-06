#encoding=utf8
"""
http://stackoverflow.com/questions/100003/what-is-a-metaclass-in-python
"""

from ipdb import set_trace as st

def make_hook(f):
    """
    Decorator to turn 'foo' method into '__foo__'
    """
    f.is_hook = 1
    return f


class MyType(type):
    def __new__(cls, name, bases, attrs):
        if name.startswith('None'):
            return None
        # Go over attributes and see if they should be renamed
        new_attrs = {}
        for attr_name, attr_value in attrs.iteritems():
            if getattr(attr_value, 'is_hook', 0):
                new_attrs['__%s__'] = attr_value
            else:
                new_attrs[attr_name] = attr_value
        return super(MyType, cls).__new__(cls, name, bases, new_attrs)

    def __init__(self, name, bases, attrs):
        super(MyType, self).__init__(name,bases, attrs)
        print "Would register class %s now." % self

    def __add__(self, other):
        class AutoClass(self, other):
            pass
        return AutoClass

    def unregister(self):
        print "Would unregister class % now." % self

class MyOjbect:
    __metaclass__ = MyType

class NoneSample(MyOjbect):
    pass

print type(NoneSample), repr(NoneSample)

class Example(MyOjbect):
    def __init__(self, value):
        self.value = value
    @make_hook
    def add(self, other):
        return self.__class__(self.value + other.value)


Example.unregister()
inst = Example(10)

print inst + inst

class Sibling(MyOjbect):
    pass
ExampleSibling = Example + Sibling

print ExampleSibling
print ExampleSibling.__mro__

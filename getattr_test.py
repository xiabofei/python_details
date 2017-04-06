#encoding=utf8


from ipdb import set_trace as st

class foo(object):

    def __init__(self):
        self.a = 'a'

    def __getattr__(self, attribute):
        return "You asked for %s, but I'm giving you default" % attribute

    def __getattribute__(self, *args, **kwargs):
        print "__getattribute__() is called"
        tmp = super(foo, self)
        return tmp.__getattribute__(*args, **kwargs)

st(context=21)
bar = foo()
print bar.a
print bar.b
print getattr(bar, 'a')
print getattr(bar, 'b')

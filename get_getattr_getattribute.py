#encoding=utf8

class C(object):
    a = 'abc'
    def __getattribute__(self, *args, **kwargs):
        print("__getattribute__() is called")
        return object.__getattribute__(self, *args, **kwargs)

    def __getattr__(self, name):
        print("__getattr__() is called")
        return name + " from get attr"

    def __get__(self, instance, owner):
        print("__get__() is called", instance, owner)
        return self

    def foo(self, x):
        print(x)

class C2(object):
    d = C()


if __name__ == '__main__':
    c = C()
    c2 = C2()
    print(c.a)
    print '\n'
    print(c.zzzz)
    print '\n'
    c2.d
    print '\n'
    print(c2.d.a)
    print '\n'
    setattr(c, "newattr", 123)
    print c.newattr


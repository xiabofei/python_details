#encoding=utf8

def f1():
    x1 = 'f1'
    print("in f1 x1 = %s" % x1)
    def f2():
        x1 = 'f1 defined in f2'
        global x2
        x2 = 'f2'
        print("in f2 x1 = %s, x2=%s" %(x1,x2))

        def f3():
            x3 = 'f3'
            print("in f3 x1 = %s, x2=%s, x3=%s" %(x1,x2,x3))

        return f3()
    f2()
    print("in f1, again: x1=%s, x2=%s" %(x1,x2))

f1()

#encoding=utf8
import sys
import gc

def make_cycle():
    l = {}
    l[0] = l

def main():
    collected = gc.collect()
    print "Garbage collector: collected %d objects." % (collected)
    print "Creating cycles..."
    for i in xrange(10):
        make_cycle()
    collected = gc.collect()
    print "Garbage collector: collected %d objects." % (collected)

ret = main()
sys.exit(ret)

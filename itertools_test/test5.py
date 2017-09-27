# encoding=utf8

from itertools import islice

class fib(object):
    def __init__(self):
        self.prev = 0
        self.curr = 1 

    def __iter__(self):
        return self

    def next(self):
        value = self.curr
        self.curr += self.prev
        self.prev = value 
        return value


f = fib()
print(list(islice(f, 0, 10)))


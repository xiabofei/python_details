#encoding=utf8
"""
werkzeug/local.py
Local
def pop
试验stack = getattr(self._local, 'stack', None)如果返回list
是返回的值还是引用
getattr返回的是list本身的地址
"""
from ipdb import set_trace

class Test(object):

    def __init__(self):
        self.stack = []

    def pop(self):
        stack = getattr(self, 'stack')
        if stack is None:
            return None
        elif len(stack) == 1:
            print stack[-1]
            self.stack.pop()
            return stack[-1]
        else:
            return stack.pop()

set_trace()
t = Test()
t.stack.append(1)
t.stack.append(2)
print t.pop()
print t.pop()


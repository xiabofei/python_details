#encoding=utf8

"""
看python2.7/socket.py中有StringIO的内容
看下面的blog熟悉StringIO的用法
以及为什么不用list, 具体代码是下面这段
   263         # We use StringIO for the read buffer to avoid holding a list
   264         # of variously sized string objects which have been known to
   265         # fragment the heap due to how they are malloc()ed and often
   266         # realloc()ed down much smaller than their original allocation.
   267         self._rbuf = StringIO()
   268         self._wbuf = [] # A list of strings
http://blog.csdn.net/zhaoweikid/article/details/1656226
"""
import StringIO

s = StringIO.StringIO()
s.write("aaaa")
lines = ['xxxxx', 'bbbbbbb']
s.writelines(lines)

s.seek(0)
print s.read()

print s.getvalue()
s.write("tttttttt")
s.seek(0)
print s.readlines()
print s.len


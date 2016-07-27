"""
http://www.cnblogs.com/ArtsCrafts/archive/2013/04/24/Python.html
threading.Event的初级用法:
    多个线程共同通等待一个线程执行完 再往下进行动作
"""
#encoding=utf8
import threading
import time

class TestThread(threading.Thread):
    def __init__(self, name, event):
        super(TestThread, self).__init__()
        self.name = name
        self.event = event
    def run(self):
        print 'Thread: ', self.name, ' start at:', time.ctime(time.time())
        self.event.wait()
        print 'Thread: ', self.name, ' finish at:', time.ctime(time.time())

# main线程定义Event变量
event = threading.Event() 
threads = []
for i in xrange(1, 3):
    # 每个线程初始化时候都传入同一个threading.Event对象
    threads.append(TestThread(str(i), event))
print 'main thread start at: ', time.ctime(time.time())
event.clear()
for thread in threads:
    thread.start()
print 'sleep 5 seconds......'
time.sleep(5)
print 'now awake other threads....'
event.set()

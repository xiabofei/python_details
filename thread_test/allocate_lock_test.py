#encoding=utf8
"""
学习thread模块中allocate_lock的用法
参考
http://blog.csdn.net/winterto1990/article/details/47845369
首先明确一点allocate_lock()返回的是一个binary flag全局变量
同一个线程不能连续两次执行acquire() 除非之前已经执行了一次 release()
可以看成allocate_lock()返回的是一个最小值是0 最大值是1的全局同步变量
这个变量初始化是0 每次执行acquire就是加1 每次执行release就减1 而且只能取值0或1 这样可以更容易理解一些

分析下面代码的执行流程
(1) main线程中 : w_ok获得锁
(2) main线程中 : 启动h线程 w线程
    由于main线程中获得了w_ok锁 所以w线程执行阻塞
    而h_ok没有上锁 所以h线程可以执行
(3) h线程中 : 先获得了h_ok锁 输出hello 再释放w_ok锁
    这时被阻塞的w线程获得了执行权
    与此同时 h线程一定不会输出下一个hello 因为h线程此时已经占用了h_ok.acquire()
    当i=1要第二次执行h_ok.acquire()时就会被阻塞住
(4) w线程中 : 占用了w_ok这个锁 输出world 并释放h_ok锁
    这时被阻塞的h进程获得了执行权 可以再次h_ok.acquire()
交替执行(3) (4)两个步骤 可以交替输出hello&world
"""

import thread

h_ok = thread.allocate_lock()
w_ok = thread.allocate_lock()

def h():
    for i in xrange(5):
        h_ok.acquire()
        print "hello"
        w_ok.release()

def w():
    for i in xrange(5):
        w_ok.acquire()
        print "world"
        h_ok.release()
print "use two threads to print hello&word"
w_ok.acquire()
thread.start_new_thread(h,())
thread.start_new_thread(w,())
raw_input("finish\n")

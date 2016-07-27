"""
http://www.cnblogs.com/ArtsCrafts/archive/2013/04/24/Python.html

"""
#encoding=utf8
import threading
import random
import time

class VehicleThread(threading.Thread):
    """
    Class repsent a motor vehicle at an intersection
    """
    def __init__(self, threadName, event):
        threading.Thread.__init__(self, name=threadName)
        self.threadEvent = event
    def run(self):
        time.sleep(random.randrange(1,10))
        print "%s arrived at %s" % \
                (self.getName(), time.ctime(time.time()))

greenLight = threading.Event()
VehicleThreads = []

for i in xrange(1, 5):
    VehicleThreads.append(VehicleThread(str(i), greenLight))

for Vechile in VehicleThreads:
    Vechile.start()

while threading.activeCount()>1:
    greenLight.clear()
    print "RED LIGHT! at", time.ctime(time.time())
    time.sleep(3)
    print "GREEN LIGHT! at", time.ctime(time.time())
    greenLight.set()
    time.sleep(1)


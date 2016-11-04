#encoding=utf8
# http://mp.weixin.qq.com/s?__biz=MzA4MjEyNTA5Mw==&mid=2652564270&idx=2&sn=92ce153690b71e6b4ca86917ecb82951&chksm=8464c364b3134a72bfc461c78fe164e21fb701f4d57d58ded47bc291af572c0a73976e020e28&scene=0#wechat_redirect
# python progress bar example

from __future__ import division

import sys, time
from progressbar import *
total = 1000

pbar = ProgressBar().start()
for i in range(1,1000):
    pbar.update(int((i/(total-1))*100))
    time.sleep(0.01)
pbar.finish()

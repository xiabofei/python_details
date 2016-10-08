# encoding=utf8
# 参考blog学习cookie原理
# https://cuiqingcai.com/968.html
import urllib2
import cookielib

cookie = cookielib.CookieJar()
handler = urllib2.HTTPCookieProcessor(cookie)
opener = urllib2.build_opener(handler)
repsonse = opener.open('http://www.baidu.com')

for item in cookie:
    print 'Name = '+item.name
    print 'Value = '+item.value

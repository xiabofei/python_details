#encoding=utf8
"""
client
"""
import socket
from ipdb import set_trace as st

messages = ["\"This is the message\"", 
        "\"It will be sent\"",
        "\"in parts\""]

print "Connect to the server"

server_adderss = ("localhost", 5001)

# 3个socket链接服务端
socks = []
for i in xrange(100):
    socks.append(socket.socket(socket.AF_INET, socket.SOCK_STREAM))    
for s in socks:
    s.connect(server_adderss)

counter = 0
for message in messages:
    for s in socks:
        counter += 1
        print " %s sending %s" % (s.getpeername(), message+" version "+str(counter))
        s.send(message+" version "+str(counter))
    for s in socks:
        data = s.recv(1024)
        print " %s receive %s " % (s.getpeername(), data)
        if not data:
            print " closing socket ",s.getpeername()
            s.close()

#encoding=utf8
"""
聊天室服务端程序
http://www.cnblogs.com/hazir/p/python_chat_room.html
目的是熟悉server端的select编程基本流程 方便看懂Flask的源码
"""
import socket
import sys
import select

# 服务端地址
HOST = 'localhost'
PORT = 5005
# 监听队列长度
LISTEN = 10
# 聊天室成员
connected_clients = []

def broadcast_all(data, client, server):
    for c in connected_clients:
        if (c not in [client, server]):
            c.sendall(data)
            
if __name__ == '__main__':
    # 创建服务端监听socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 在bind前设置端口复用
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PORT))
    server_socket.listen(LISTEN)
    print 'Server listen in port %s ' % (PORT)
    connected_clients.append(server_socket)
    while 1:
        r, w, e = select.select(connected_clients, [], [], 20)
        for s in r:
            if s is server_socket:
                conn, addr = server_socket.accept()
                connected_clients.append(conn)
            else:
                data = s.recv(4096)
                if not data:
                    connected_clients.remove(s)
                else:
                    print '<(%s, %s)> : ' % addr, data
                    broadcast_all(data, s, server_socket)
        for s in w:
            pass
        for s in e:
            pass

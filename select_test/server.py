#encoding=utf8
"""
select.select服务端
原文
http://www.cnblogs.com/coser/archive/2012/01/06/2315216.html
下面的blog是比较好的解释
http://my.oschina.net/u/1433482/blog/191211
"""
import select
import socket
import Queue

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setblocking(False)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
address = ('localhost',5001)
server.bind(address)
server.listen(10) # 等待队列长度为10
inputs = [server]
outputs = []
message_queues = {}
timeout = 20

select_time = 0
while inputs:
    print "waiting for next event"
    select_time += 1
    # select的4个输入参数 ( 按顺序 )
    # 读,写,错误,超时
    # 前三个是监控的对象(每个监控的对象都是socket列表)
    # from ipdb import set_trace as st
    # st()
    r, w, e = select.select(inputs, outputs, inputs, timeout)
    if not ( r or w or e):
        print "Time out !"
        break
    for s in r:
        # 有client的连接请求
        # 1. accept这个client, 并分配一个socket
        # 2. 将这个client的socket加入select的read的list
        # 3. 给这个client分配一个消息队列Queue
        if s is server:
            connect, client_addr = s.accept()
            print "  connection from " , client_addr
            connect.setblocking(0)
            inputs.append(connect)
            message_queues[connect] = Queue.Queue()
        # 有已accept的client发来消息
        # 1. 只接受1024的data
        # 2. 还没有向client返回数据, 则将client的socket加入select的write的list
        # 3. 如果发来消息为空, 则直接将client的socket从select的read和write中剔除
        else:
            data = s.recv(1024)
            if data:
                print " received " , data, "from ", s.getpeername()
                message_queues[s].put(data)
                if s not in outputs:
                    outputs.append(s)
            else:
                print " closing", client_addr
                if s in outputs:
                    outputs.remove(s)
                inputs.remove(s)
                s.close()
                del message_queues[s]
    # 如果某个client的Queue中有数据, 则把数据send回去
    # 如果某个client的Queue中没有数据, 则把这个client从select的write清除 
    for s in w:
        try:
            next_msg = message_queues[s].get_nowait()
        except Queue.Empty:
            print " ", s.getpeername(), 'queue empty'
            outputs.remove(s)
        else:
            print " sending ", next_msg, " to ", s.getpeername()
            s.send(next_msg)
    # 如果client有异常, 则直接删除掉
    for s in e:
        print " exception condition on ", s.getpeername()
        inputs.remove(s)
        if s in outputs:
            outputs.remove(s)
        s.close()
        del message_queues[s]

print select_time

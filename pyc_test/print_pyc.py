# encoding=utf8
# http://www.restran.net/2015/10/22/how-python-code-run/
# 将code对象中的字节码指令输出
# 输出'字节码指令'信息 这里是'指令'而不是二进制文件

s = open('demo.py').read()
co = compile(s, 'demo.py', 'exec')
import dis
dis.dis(co)

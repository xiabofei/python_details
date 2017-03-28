#encoding=utf8
"""
生成器
fibonacci()通过yield返回的是一个iterator对象
"""

def fibonacci():
    a, b = 0, 1
    while True:
        yield b
        a, b = b, a+b

fib = fibonacci()

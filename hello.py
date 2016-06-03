#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'Michael Liao'

import sys

def test():
    args = sys.argv
    if len(args) == 1:
        print 'Hello, world!'
    elif len(args) == 2:
        print 'Hello, %s!' % args[1]
    else:
        print 'too many arguments!'

# 运行这个.py文件时 python解释器把一个特殊变量__name__置为__main__
# if __name__ == '__main__':
    # test()

test()

"""
http://gold.xitu.io/entry/575bdbe8207703006ffb2b00
python解释器搜索modules的顺序:
    1) 已经在sys.modules中加载的modules
    2) 当前目录
    3) 第三方包的路径
这样的优先级会带来一些同名module文件的覆盖问题
因此, 根据优先级built-in的os和sys都会引入
即使当前目录下也有os.py和sys.py的module
但是flask.py是第三方module, 优先引入当前目录的文件
"""
from os import path
from sys import modules
from flask import Flask

#encoding=utf8
# 生成字节码.pyc文件
# 再执行完imp.load_module(name, fp, pathname, description)
# 这条load语句后 会生成.pyc文件
# http://www.restran.net/2015/10/22/how-python-code-run/

import imp
import sys
from ipdb import set_trace as st

def generate_pyc(name):
    st(context=21)
    fp, pathname, description = imp.find_module(name)
    print fp
    print pathname
    print description
    try:
        imp.load_module(name, fp, pathname, description)
    finally:
        if fp:
            fp.close()

if __name__ == '__main__':
    generate_pyc(sys.argv[1])

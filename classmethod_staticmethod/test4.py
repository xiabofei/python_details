#encoding=utf8


class FormMeta(type):
    def __init__(cls):
        print '__init__:'+str(cls)
    def __new__(cls):
        print '__new__:'+str(cls)
    def __call__(cls):
        print '__call__:'+str(cls)


class Form(object):
    def __init__(self):
        print '__init__:'+str(self)
    def __new__(self):
        print '__new__:'+str(self)
    def __call__(self):
        print '__call__:'+str(self)


from ipdb import set_trace as st
st(context=21)
meta = FormMeta()
instance = Form()

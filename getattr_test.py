#encoding=utf8

class foo:
    def __init__(self):
        self.a = 'a'
    def __getattr__(self, attribute):
        return "You asked for %s, but I'm giving you default" % attribute

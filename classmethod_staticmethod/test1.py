#encoding=utf8
"""
https://julien.danjou.info/blog/2013/guide-python-static-class-abstract-methods
"""

# instancemethod
class Pizza1(object):
    def __init__(self, size):
        self.size = size
    def get_size(self):
        return self.size

# staticmethod
class Pizza2(object):
    @staticmethod
    def mix_ingredients(x, y):
        return x + y
    def cook(self):
        return self.mix_ingredients(self.cheese, self.vegetables)

# classmethod
class Pizza3(object):
    radius = 42
    @classmethod
    def get_radius(cls):
        return cls.radius

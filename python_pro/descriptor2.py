#encoding=utf8

"""
"""
import random

class Die(object):
    def __init__(self, sides=6):
        self.sides = sides
    def __get__(self, instance, owner):
        return int(random.random()*self.sides) + 1

class Game(object):
    d6 = Die()
    d10 = Die(sides=10)
    d20 = Die(sides=20)




#encoding=utf8
"""
http://stackoverflow.com/questions/12179271/meaning-of-classmethod-and-staticmethod-for-beginner
这个例子说明了classmethod的一种简单用法 可以根据subclass的属性不同采取不同的处理方式
"""

class Hero:
    @staticmethod
    def say_hello():
        print "hello..."

    @classmethod
    def say_class_hello(cls):
        if cls.__name__=="HeroSon":
            print "Hi Kido"
        elif cls.__name__=="HeroDaughter":
            print "Hi Princess"

class HeroSon(Hero):
    def say_son_hello(self):
        print "test hello son"

class HeroDaughter(Hero):
    def say_daughter_hello(self):
        print "test hello daughter"
            
test_son = HeroSon()
test_son.say_class_hello()
test_son.say_hello()

test_daughter = HeroDaughter()
test_daughter.say_class_hello()
test_daughter.say_hello()

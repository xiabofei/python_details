# encoding=utf8

# 国内的大部分都不太对 看这个
# https://www.python-course.eu/python3_properties.php
# py2要想用property 需要继承object / py3不用继承object
# 在property里面定义一个class private私有变量

class Student:

    def __init__(self, val):
        self.age = val # 在init之前age的set和get已经加载好了, 因此也会调用age.setter去检验

    @property
    def age(self):
        print('getter')
        return self.__age

    @age.setter
    def age(self, val):
        print('setter')
        if val>90:
            raise ValueError('too large age : %s' % val)
        self.__age = val

    def other_age(self):
        print(self.age)
        self.age = 100


if __name__ == '__main__':
    s = Student(60)
    print(s.age)
    s.age = 10
    print(s.age)
    s.other_age()





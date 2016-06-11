#encoding=utf8

"""
学习yield用法
http://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do-in-python
"""

def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator()
print(mygenerator)

for i in mygenerator:
    print(i)

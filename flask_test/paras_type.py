#encoding=utf8

"""
werkzeug/local.py 
Local
__getattr__中用到的dict类型的名传递
(1) dict name是地址传递
(2) list name也是地址传递
(3) tuple中的内容不能修改 (新增加的知识)
"""
def test_dict_name(name):
    name[1] = 1

def test_list_name(name):
    name[0] =1

def test_tuple_name(name):
    name[0] = 1

dict_name = {}
print dict_name
test_dict_name(dict_name)
print dict_name

list_name = []
print list_name
list_name.append('a')
test_list_name(list_name)
print list_name

tuple_name = ("abc")
print tuple_name
test_tuple_name(tuple_name) # 会报错
print tuple_name

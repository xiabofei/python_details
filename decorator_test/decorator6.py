#encoding=utf8

"""
装饰有返回值的函数
"""

def print_debug(func):
    def __decorator(user):
        print('enter the login')
        result = func(user)
        print('exit the login')
        return result
    return __decorator

@print_debug
def login(user):
    print('in login:' + user)
    msg = "success" if user == "jatsz" else "fail"
    return msg

result1 = login('jatsz')
print result1

result2 = login('candy')
print result2

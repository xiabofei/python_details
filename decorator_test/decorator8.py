#encoding=utf8

"""
实际例子 用decorator方法 在不修改login函数的情况下
增加cache来提升登录速度
"""

import time

dictcache = {}

def cache(func):
    def __decorator(user):
        now = time.time()
        if user in dictcache:
            result, cache_time = dictcache[user]
            if now - cache_time > 30:
                result = func(user)
                dictcache[user] = (result, now)
            else:
                print('cache hits')
        else:
            result = func(user)
            dictcache[user] = (result, now)
        return result
    return __decorator

def login(user):
    print('in login:' + user)
    msg = validate(user)
    return msg

@cache
def validate(user):
    time.sleep(5)
    msg = "success" if user == "jatsz" else "fail"
    return msg

print login('jatsz')
print login('jatsz')
print login('candy')

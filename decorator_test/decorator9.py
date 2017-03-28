#encoding=utf8

import json
import functools

def json_output(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        return json.dumps(result)
    return inner


@json_output
def f():
    return {'status': 'done'}

#encoding=utf8
"""
islice可以用于每次都从某个stream的固定的位置抽取数据
"""

import itertools

def starting_at_five():
    value = raw_input().strip()
    while value != '':
        for el in itertools.islice(value.split(), 4, None):
            yield el
        value = raw_input().strip()

# coding=utf8

import re


def remove_digit(content):
    return re.sub("\d*\.", "", content)


print remove_digit("1111.糖尿病")

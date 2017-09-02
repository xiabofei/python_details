# coding=utf8

import re
import codecs

import sys

reload(sys)
sys.setdefaultencoding('utf-8')

from ipdb import set_trace as st

CLRF = '\n'

def multiple_replace(text, adict):
    rx = re.compile('|'.join(map(re.escape, adict)))
    def one_xlat(match):
        return adict[match.group(0)]
    return rx.sub(one_xlat, text)

# 去掉数字和小数点
def remove_digits(text):
    pattern = re.compile("[0-9\.]*")
    return re.sub(pattern, "", text)

# 去掉单位,括号内容
def remove_units(text):
    parenthesis = re.compile("\(.*?\)")
    units1 = re.compile("[0-9\.]+[A-Za-z]+\[A-Za-z]+")
    units2 = re.compile("[0-9\.]+[A-Za-z]+")

    text = re.sub(parenthesis, "", text)
    text = re.sub(units1, "", text)
    text = re.sub(units2, "", text)
    return text

# 将连续的标点换成一个逗号
def combine_punc(text):
    pattern = re.compile("[.,:;\"\"?]+")
    return re.sub(pattern, ",", text)

# 去掉标题(一,二, ...)
def remove_subtitle(text):
    pattern = re.compile(u"[一二三四五]+,")
    return re.sub(pattern, "", text)

def remove_info(text):
    name = re.compile(u",(.*?)姓名(.*?),")
    gender = re.compile(u",(.*?)性别(.*?),")
    age = re.compile(u",(.*?)年龄(.*?),")
    text = re.sub(name, "", text)
    text = re.sub(gender, "", text)
    text = re.sub(age, "", text)
    return text


if __name__ == '__main__':
    # print remove_info(u",患者姓名a,年龄b")
    replace_punc = {
        u"，": ",",
        u"。": ".",
        u"；": ";",
        u"：": ":",
        u"？": "?",
        u"“": u"\"",
        u"”": u"\"",
        u"、": ",",
        u"／": "/",
        u"（": "(",
        u"）": ")",
        "\t": "",
        "\n": "",
    }
    replace_unicode = {
        u"次/分": "",
        u"次/次分": "",
        u"℃": "",
        u"病例特点":""
    }

    replace_unit = {
        "-": "",
        "%": "",
        "/": "",
        "*": "",
        "^": "",
        "&;": "",
        " ": "",
        "．":"",
        "&quot":"",
        "°":"",
        "C":""
    }

    keshi = set()
    with open("../data/output/extracted/keshi_all.dat") as file_keshi:
        for line in file_keshi.readlines():
            keshi.add(line.decode('utf8').replace("\n", ""))
    with open("../data/output/extracted/all_extracted.dat") as file_all_data:
        lines = file_all_data.readlines()

    with codecs.open("../data/output/merged/all_merged.csv", "w", encoding="utf8") as new_data:
        new_data.write(u"patient_id\tkeshi\tzhenduan\tzhusu\txianbingshi\tjiwangshi\tgerenshi\tbingchengjilu\n")
        last_patient_id = ""
        cur_patient_info = [''] * 8
        for line in lines:
            line = line.decode('utf8')
            l = line.split("\t")
            cur_patient_id, type_id, info = l[0].strip(), int(l[1].strip()), l[2].strip()
            if cur_patient_id != last_patient_id and last_patient_id:
                new_data.write('\t'.join(cur_patient_info) + CLRF)
                cur_patient_info = [''] * 8
            last_patient_id = cur_patient_id
            cur_patient_info[0] = cur_patient_id
            if info:
                if type_id==0 and info not in keshi:
                    info = ''
                else:
                    replaced_text = multiple_replace(info, replace_punc)
                    replaced_text = multiple_replace(replaced_text, replace_unicode)
                    replaced_text = multiple_replace(replaced_text, replace_unit)
                    replaced_text = remove_units(replaced_text)
                    replaced_text = remove_digits(replaced_text)
                    replaced_text = remove_subtitle(replaced_text)
                    replaced_text = combine_punc(replaced_text)
                    info = replaced_text
            cur_patient_info[type_id+1] += info
        if last_patient_id:
            new_data.write('\t'.join(cur_patient_info) + CLRF)

# encoding=utf8

import jieba
import jieba.posseg as poseg
import pandas as pd

from ipdb import set_trace as st

JOIN = '/'
CLRF = '\n'

input_path = './data/input/MRI.csv'
output_path = './data/output/'

columns = ['ExamDescExResult', 'ResultDescExResult']

usr_dict_path = './data/output/mri_dict.dat'


def _add_usr_dict(path):
    with open(path, 'r') as f:
        for l in f.xreadlines():
            word = l.strip()
            jieba.add_word(word, 50)


def _replace_punctuation(content):
    _chinese_english = [
        ('，', ','),
        ('、', ','),
        ('（', '('),
        ('）', ')'),
        ('。', '.'),
        ('；', ';'),
        ('：', ':'),
        ('“', '"'),
        ('－', '-'),
        ("”", '"')
    ]
    for i in _chinese_english:
        content = content.replace(i[0], i[1])
    return content


def _negative_positive(content):
    _symbol_word = [
        ('(-)', '阴性'),
        ('(+)', '阳性'),
    ]
    for i in _symbol_word:
        content = content.replace(i[0], i[1])
    return content


def _cut(content):
    return jieba.cut(content)


def pre_process(content):
    _process_flow = [_replace_punctuation, _negative_positive]
    ret = content
    for flow in _process_flow:
        ret = flow(ret)
    return ret


with open(input_path, 'r') as f_input:
    df = pd.read_csv(f_input)
    _add_usr_dict(usr_dict_path)
    for column in columns:
        with open(output_path + column + '_with_usr_dict', 'w') as f_output:
            for i in df.index:
                content = pre_process(df.loc[i, column])
                seg_list = _cut(content)
                seg_list = [seg.encode('utf8') for seg in seg_list]
                f_output.write(JOIN.join(seg_list) + CLRF)
                for k, v in poseg.cut(content):
                    f_output.write(k.encode('utf8') + ',' + v.encode('utf8') + CLRF)

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

usr_dict_path = './data/input/mri_dict.dat'


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
        (' ', ''),
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
                f_output.write('[原文]:' + CLRF)
                content = pre_process(df.loc[i, column])
                f_output.write(content + CLRF)
                # seg_list = _cut(content)
                # seg_list = [seg.encode('utf8') for seg in seg_list]
                # f_output.write(JOIN.join(seg_list) + CLRF)
                f_output.write('[词性标注结果]:' + CLRF)
                for con in content.split(';'):
                    f_output.write('\t' + ''.join(
                        [k.encode('utf8') + '/' + v.encode('utf8') + ' ' for k, v in poseg.cut(con)]) + CLRF)
                f_output.write(CLRF)

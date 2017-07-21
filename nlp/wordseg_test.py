# encoding=utf8

import jieba
import jieba.posseg as poseg
import pandas as pd
from collections import OrderedDict

from ipdb import set_trace as st

JOIN = '/'
CLRF = '\n'

input_path = './data/input/MRI.csv'
output_path = './data/output/'

columns = ['ExamDescExResult', 'ResultDescExResult']

usr_dict_path = './data/input/mri_dict.dat'

usr_suggest_path = './data/input/mri_suggest.dat'


def _add_usr_dict(path):
    with open(path, 'r') as f:
        for l in f.xreadlines():
            items = l.split(',')
            jieba.add_word(items[0].rstrip(), items[1].rstrip(), items[2].rstrip())

def _suggest_usr_dict(path):
    with open(path, 'r') as f:
        for l in f.xreadlines():
            word1, word2 = l.split(',')[0].rstrip(), l.split(',')[1].rstrip()
            jieba.suggest_freq((word1, word2), True)

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


def _clear_trivial_head(content):
    pos = content.find('复查')
    if pos >= 0:
        return content[pos + len('复查'):]
    else:
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


def _find_topic_range(content, topics):
    _legal_punctuation = (':',)
    _pos_punctuation = (',', ';', '.',)
    ret = []
    for topic in topics:
        pos = content.find(topic)
        if pos >= 0:
            if content[pos + len(topic):pos + len(topic) + 1] not in _legal_punctuation:
                _nearest_right_comma_pos = max([content[:pos].rfind(p) for p in _pos_punctuation]) + 1
                ret.append([_nearest_right_comma_pos, topic])
            else:
                ret.append([pos, topic])
    ret = sorted(ret, key=lambda x: x[0])
    for i in range(len(ret) - 1):
        ret[i][0] = (ret[i][0], ret[i + 1][0])
    ret[len(ret) - 1][0] = (ret[len(ret) - 1][0], len(content))
    return ret


def pre_process(content, process_flow):
    ret = content
    for flow in process_flow:
        ret = flow(ret)
    return ret


# _mri_topic = ('宫体', '宫颈', '阴道及外阴', '双侧附件', '淋巴结', '其他', '盆腔积液', '骨盆骨质')



_deny_word = (u'不伴', u'未见', u'未', u'无',)

_ensure_word = (u'可见', u'伴',)


def extract_type_one_words(topic_content):
    words = ('T1', 'T2')
    ret = []
    _des_pattern = ('n+v', 'v+n', 'u+n', 'd+l', 'a+n', 'n+ns', 'n+a')
    _degree_pos = ('u', 'a', 'm')
    for index, tc in enumerate(topic_content):
        if tc[0] in words and index > 0:
            pre = []
            for i in range(index - 1, -1, -1):
                if topic_content[i][1] == 'f':
                    pre.append(topic_content[i][0])
                    for j in range(i - 1, -1, -1):
                        if topic_content[j][1] == 'f':
                            pre.append(topic_content[j][0])
                        elif topic_content[j][1] == 'x':
                            break
                    break
            des = []
            for i in range(index + 1, len(topic_content) - 1):
                if topic_content[i][1] == 'x' or topic_content[i][0] in words:
                    break
                pattern = topic_content[i][1] + '+' + topic_content[i + 1][1]
                if pattern in _des_pattern:
                    des.append(topic_content[i][0])
                    des.append(topic_content[i + 1][0])
            degree = []
            if topic_content[index - 1][1] in _degree_pos:
                degree.append(topic_content[index - 1][0])
            degree = '_'.join(degree)
            if len(degree) > 0:
                if len(des) > 0:
                    des.insert(0, degree)
                else:
                    des.append(degree)
            neg_or_pos = u'有'
            for i in range(index - 1, -1, -1):
                if topic_content[i][1] == 'x':
                    break
                if topic_content[i][0] in _deny_word:
                    neg_or_pos = u'无'
                    break
            ret.append(
                '区域:' + '_'.join(reversed(pre)).encode('utf-8') + ', ' +
                '指标:' + tc[0].encode('utf-8') + ', ' +
                '有无:' + neg_or_pos.encode('utf-8') + ', ' +
                '描述:' + '_'.join(des).encode('utf-8')
            )
    return ret


def extract_type_two_words(topic_content):
    words = (u'结合带',)
    ret = []
    for index, tc in enumerate(topic_content):
        if tc[0] in words:
            _tmp = []
            for i in range(index + 1, len(topic_content)):
                if 'x' not in topic_content[i][1]:
                    _tmp.append(topic_content[i][0])
                else:
                    break
            ret.append('指标:' + tc[0].encode('utf-8') + ',' + '结果:' + (''.join(_tmp)).encode('utf-8'))
    return ret


def extract_type_three_words(topic_content):
    words = (u'盆腔积液',)
    ret = []
    for index, tc in enumerate(topic_content):
        if tc[0] in words:
            has_or_not = '有'
            # nearest after
            for i in range(index + 1, len(topic_content)):
                if topic_content[i][1] == 'x':
                    break
                if topic_content[i][0] in _deny_word:
                    has_or_not = '无'
                    break
            # nearest before
            for i in range(index, -1, -1):
                if topic_content[i][1] == 'x':
                    break
                if topic_content[i][0] in _deny_word:
                    has_or_not = '无'
                    break
            ret.append('指标:' + tc[0].encode('utf-8') + ',' + '有无:' + has_or_not)
            break
    return ret


_mri_topic_extract_methods = OrderedDict([
    ('宫体', [extract_type_one_words, extract_type_two_words]),
    ('宫颈', [extract_type_one_words, extract_type_two_words]),
    ('阴道及外阴', [extract_type_one_words, extract_type_two_words]),
    ('双侧附件', [extract_type_one_words, extract_type_two_words]),
    ('淋巴结', [extract_type_one_words, extract_type_two_words]),
    ('其它', [extract_type_one_words, extract_type_two_words]),
    ('盆腔积液', [extract_type_one_words, extract_type_three_words]),
    ('骨盆骨质', []),
])

with open(input_path, 'r') as f_input:
    df = pd.read_csv(f_input)
    # st(context=21)
    _add_usr_dict(usr_dict_path)
    _suggest_usr_dict(usr_suggest_path)
    with open(output_path + 'mri_describe_result', 'w') as f_output:
        for i in df.index:
            # show origin mri report text
            content = pre_process(df.loc[i, columns[0]],
                                  [_replace_punctuation, _negative_positive, _clear_trivial_head])
            result = pre_process(df.loc[i, columns[1]],
                                 [_replace_punctuation, _negative_positive])
            f_output.write('[报告原文-全]:' + CLRF)
            f_output.write(content + CLRF)
            f_output.write(CLRF + '[报告解读]:' + CLRF)
            f_output.write(result + CLRF)
            # show word seg result
            topic_range = _find_topic_range(content, _mri_topic_extract_methods.keys())
            for item in topic_range:
                begin, end, topic = item[0][0], item[0][1], item[1]
                f_output.write(CLRF + "[报告原文-%s]" % (topic) + CLRF)
                f_output.write(content[begin:end] + CLRF)
                topic_content = [(k, v) for k, v in poseg.cut(content[begin:end], HMM=False)]
                f_output.write('[信号提取结果-%s]:' % (topic) + CLRF)
                for method in _mri_topic_extract_methods[topic]:
                    f_output.write(CLRF.join(method(topic_content)) + CLRF)
                f_output.write("[分词结果-%s]" % (topic) + CLRF)
                f_output.write('\t' + ''.join(
                    [i[0].encode('utf8') + '/' + i[1].encode('utf8') + ' ' for i in topic_content]) + CLRF)
            f_output.write('=' * 80 + CLRF)

# encoding=utf8

import jieba.posseg as poseg
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from utils.prepare_utils import JiebaTuning as JT
from utils.prepare_utils import GeneralAddressing as GA
import json

from ipdb import set_trace as st

JOIN = '/'
CLRF = '\n'


class XHLanguageTemplate(object):
    _deny_word = (u'不伴', u'未见', u'未', u'无', u'不', u'未见异常',)

    _ensure_word = (u'可见', u'伴', u'为', u'呈')

    @classmethod
    def one(cls, topic_content):
        words = ('T1', 'T2')
        ret = defaultdict(dict)
        num = 1
        _des_pattern = ('n+v', 'v+n', 'u+n', 'd+l', 'a+n', 'n+ns', 'f+v+a', 'n+a', 'd+n', 'n+n+v+a+v',)
        _des_pattern_split = '+'
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

                degree = []
                if topic_content[index - 1][1] in _degree_pos:
                    degree.append(topic_content[index - 1][0])
                degree = '_'.join(degree)

                des = []
                i = index + 1
                while i < len(topic_content) - 1:
                    if topic_content[i][1] == 'x' or topic_content[i][0] in words:
                        if degree != '' or len(des) != 0:
                            break
                    not_match = True
                    for pattern in _des_pattern:
                        items = pattern.split(_des_pattern_split)
                        if i + len(items) > len(topic_content):
                            continue
                        _real_pattern = _des_pattern_split.join([topic_content[l][1] for l in range(i, i + len(items))])
                        if _real_pattern == pattern:
                            des.append(''.join([topic_content[l][0] for l in range(i, i + len(items))]))
                            i += len(items)
                            not_match = False
                            break
                    if not_match:
                        i += 1
                    else:
                        break

                if len(degree) > 0:
                    if len(des) > 0:
                        des.insert(0, degree)
                    else:
                        des.append(degree)

                neg_or_pos = u'有'
                for i in range(index - 1, -1, -1):
                    if topic_content[i][1] == 'x' and i != (index - 1):
                        break
                    if topic_content[i][0] in cls._deny_word:
                        neg_or_pos = u'无'
                        break
                ret[num]['检查子区域'] = '_'.join(reversed(pre)).encode('utf-8')
                ret[num]['指标'] = tc[0].encode('utf-8')
                ret[num]['断言'] = neg_or_pos.encode('utf-8')
                ret[num]['描述'] = '_'.join(des).encode('utf-8')
                num += 1
        return ret

    @classmethod
    def two(cls, topic_content):
        words = (u'结合带',)
        ret = defaultdict(dict)
        num = 1
        for index, tc in enumerate(topic_content):
            if tc[0] in words:
                _tmp = []
                for i in range(index + 1, len(topic_content)):
                    if 'x' not in topic_content[i][1]:
                        _tmp.append(topic_content[i][0])
                    else:
                        break
                ret[num]['指标'] = tc[0].encode('utf-8')
                ret[num]['结果'] = (''.join(_tmp)).encode('utf-8')
                num += 1
        return ret

    @classmethod
    def three(cls, topic_content):
        words = (u'盆腔积液',)
        ret = defaultdict(dict)
        num = 1
        for index, tc in enumerate(topic_content):
            if tc[0] in words:
                has_or_not = '有'
                # nearest after
                for i in range(index + 1, len(topic_content)):
                    if topic_content[i][1] == 'x' and i > index + 1:
                        break
                    if topic_content[i][0] in cls._deny_word:
                        has_or_not = '无'
                        break
                # nearest before
                for i in range(index, -1, -1):
                    if topic_content[i][1] == 'x':
                        break
                    if topic_content[i][0] in cls._deny_word:
                        has_or_not = '无'
                        break
                ret[num]['指标'] = tc[0].encode('utf-8')
                ret[num]['断言'] = has_or_not
                num += 1
                break
        return ret

    @classmethod
    def four(cls, topic_content):
        words = (u'DWI',)
        ret = defaultdict(dict)
        num = 1
        for index, tc in enumerate(topic_content):
            if tc[0] in words:
                if_deny = '有'
                high_or_low = '高信号'
                location = []
                for i in range(index - 1, -1, -1):
                    if topic_content[i][1] == 'f':
                        location.append(topic_content[i][0])
                        for j in range(i - 1, -1, -1):
                            if topic_content[j][1] == 'f':
                                location.append(topic_content[j][0])
                            elif topic_content[j][1] == 'x':
                                break
                        break
                for i in range(index + 1, len(topic_content)):
                    if topic_content[i][1] == 'x':
                        break
                    if topic_content[i][0] in cls._deny_word:
                        if topic_content[i][1] == 'd' and i < len(topic_content) - 1 and topic_content[i + 1][1] == 'a':
                            continue
                        if_deny = '无'
                    if topic_content[i][0] == u'低信号':
                        high_or_low = '低信号'
                ret[num]['检查子区域'] = '_'.join(reversed(location)).encode('utf-8')
                ret[num]['指标'] = tc[0].encode('utf-8')
                ret[num]['断言'] = if_deny
                ret[num]['描述'] = high_or_low
                num += 1
        return ret


class XHPipeline(object):
    def __init__(self,
                 input_path, columns, output_path,
                 usr_dict_path, usr_suggest_path,
                 word_segger, topic_pipeline):
        self.input_path = input_path
        self.columns = columns
        self.output_path = output_path
        self.usr_dict_path = usr_dict_path
        self.usr_suggest_path = usr_suggest_path
        self.word_segger = word_segger
        self.topic_pipeline = topic_pipeline

    def __setattr__(self, key, value):
        if key=='word_segger':
            JT.add_usr_dict(self.usr_dict_path)
            JT.suggest_usr_dict(self.usr_suggest_path)
            object.__setattr__(self, key, value)
        object.__setattr__(self, key ,value)

    def find_topic_range(self, content, topics):
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

    def execute_pipeline(self):
        for content in self.read_mri_report():
            yield self.extract_structural_info(content[0])

    def read_mri_report(self):
        with open(self.input_path, 'r') as f_input:
            df = pd.read_csv(f_input)
            for i in df.index:
                mri_describe = GA.execute_general_addressing(
                    df.loc[i, self.columns[0]],
                    [GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head]
                )
                mri_result = GA.execute_general_addressing(
                    df.loc[i, self.columns[1]],
                    [GA.replace_punctuation, GA.negative_positive]
                )
                yield (mri_describe, mri_result)

    def extract_structural_info(self, content):
        structural_info = {}
        topic_range = self.find_topic_range(content, self.topic_pipeline.keys())
        for item in topic_range:
            begin, end, topic = item[0][0], item[0][1], item[1]
            topic_content = [(k, v) for k, v in self.word_segger.cut(content[begin:end], HMM=False)]
            structural_info[topic] = [ template(topic_content) for template in self.topic_pipeline[topic] ]
        return structural_info

    def write_mri_structural_info(self):
        with open(self.output_path, 'w') as f_output:
            for structural_info in self.execute_pipeline():
                f_output.write(json.dumps(structural_info, indent=4, ensure_ascii=False) + CLRF)


if __name__ == '__main__':

    xiehe_topic_pipeline = OrderedDict([
        ('宫体', [XHLanguageTemplate.one, XHLanguageTemplate.two]),
        ('宫颈', [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        ('阴道及外阴', [XHLanguageTemplate.one, XHLanguageTemplate.two]),
        ('双侧附件', [XHLanguageTemplate.one, XHLanguageTemplate.two]),
        ('淋巴结', [XHLanguageTemplate.one, XHLanguageTemplate.two]),
        ('其它', [XHLanguageTemplate.one, XHLanguageTemplate.two]),
        ('盆腔积液', [XHLanguageTemplate.one, XHLanguageTemplate.three]),
    ])

    columns = ['ExamDescExResult', 'ResultDescExResult']

    usr_dict_path = './data/input/mri_dict.dat'
    usr_suggest_path = './data/input/mri_suggest.dat'

    input_path = './data/input/MRI.csv'
    output_path = './data/output/mri_describe_result.dat'

    word_segger = poseg

    xh_pipeline = XHPipeline(
        input_path, columns, output_path,
        usr_dict_path, usr_suggest_path,
        word_segger,
        xiehe_topic_pipeline
    )
    xh_pipeline.write_mri_structural_info()

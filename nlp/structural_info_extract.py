# encoding=utf8

import jieba.posseg as poseg
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from utils.prepare_utils import JiebaTuning as JT
from utils.prepare_utils import GeneralAddressing as GA
import json
import cPickle
import numpy as np

from ipdb import set_trace as st

JOIN = '/'
CLRF = '\n'


class XHLanguageTemplate(object):
    _exam_areas = (
        u'膀胱',
        u'宫体/子宫',
        u'宫颈',
        u'左附件',
        u'右附件',
        u'直肠',
        u'盆腔骨',
        u'盆腔肌肉',
        u'直肠膀胱三角',
        u'盆腔淋巴结',
        u'盆腔积液',
        u'阴道',
        u'外阴',
        u'宫旁',
        u'宫腔积液',
        u'其他'
    )

    _deny_word = (u'不伴', u'未见', u'未', u'无', u'不', u'未见异常',)

    _ensure_word = (u'可见', u'伴', u'为', u'呈')

    @classmethod
    def one(cls, topic_content):
        words = ('T1', 'T2')
        ret = []
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
                tmp_info = {}
                tmp_info['检查子区域'] = '_'.join(reversed(pre)).encode('utf-8')
                tmp_info['指标'] = tc[0].encode('utf-8')
                tmp_info['断言'] = neg_or_pos.encode('utf-8')
                tmp_info['描述'] = '_'.join(des).encode('utf-8')
                ret.append(tmp_info)
        return ret

    @classmethod
    def two(cls, topic_content):
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
                tmp_info = {}
                tmp_info['指标'] = tc[0].encode('utf-8')
                tmp_info['结果'] = (''.join(_tmp)).encode('utf-8')
                ret.append(tmp_info)
        return ret

    @classmethod
    def three(cls, topic_content):
        words = (u'盆腔积液',)
        ret = []
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
                tmp_info = {}
                tmp_info['指标'] = tc[0].encode('utf-8')
                tmp_info['断言'] = has_or_not
                ret.append(tmp_info)
                break
        return ret

    @classmethod
    def four(cls, topic_content):
        words = (u'DWI',)
        ret = []
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
                tmp_info = {}
                tmp_info['检查子区域'] = '_'.join(reversed(location)).encode('utf-8')
                tmp_info['指标'] = tc[0].encode('utf-8')
                tmp_info['断言'] = if_deny
                tmp_info['描述'] = high_or_low
                ret.append(tmp_info)
        return ret


class XHPipeline(object):
    def __init__(self,
                 input_path, columns, output_path,
                 usr_dict_path, usr_suggest_path,
                 label_data_path,
                 word_segger, topic_pipeline):
        self.input_path = input_path
        self.columns = columns
        self.output_path = output_path
        self.usr_dict_path = usr_dict_path
        self.usr_suggest_path = usr_suggest_path
        self.label_data_path = label_data_path
        self.word_segger = word_segger
        self.topic_pipeline = topic_pipeline
        self.word_frequency = defaultdict(int)

    def __setattr__(self, key, value):
        if key == 'word_segger':
            JT.add_usr_dict(self.usr_dict_path)
            JT.suggest_usr_dict(self.usr_suggest_path)
            object.__setattr__(self, key, value)
        object.__setattr__(self, key, value)

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

    def execute_pipeline(self, pipe_type):
        for content in self.read_mri_report():
            if pipe_type == 'label':
                yield self.extract_paragraph_for_label(content[0], content[1])
            elif pipe_type == 'struct':
                yield self.extract_structural_info(content[0])
            elif pipe_type == 'cut':
                yield self.create_split_text(content[0])


    def extract_paragraph_for_label(self, content, result):
        topic_range = self.find_topic_range(content, self.topic_pipeline.keys())
        paragraph = defaultdict(dict)
        paragraph['MRI报告'] = content
        paragraph['MRI解读'] = result
        for word, pos in self.word_segger.cut(content, HMM=False):
            if pos not in ('x',):
                try:
                    int(word)
                except Exception, e:
                    if 'mm' not in word and 'cm' not in word and u'的' not in word:
                        self.word_frequency[word] += 1
        paragraph['各区域标注结果'] = {area.encode('utf-8'): "" for area in XHLanguageTemplate._exam_areas}
        # paragraph['切词结果'] = ' '.join(
        #     [k.encode('utf-8') + '/' + v.encode('utf-8') for k, v in self.word_segger.cut(content, HMM=False) if
        #      v not in ('x',)])
        # for item in topic_range:
        #     begin, end, topic = item[0][0], item[0][1], item[1]
        #     paragraph[topic]['content'] = content[begin:end]
        #     paragraph[topic]['label'] = self.label_proxy(content[begin:end], 'rule')
        return paragraph

    def label_proxy(self, content, label_type):
        if label_type == 'rule':
            return self.label_by_rule(content)

    def label_by_rule(self, content):
        _negative_words = ('未见', '无异常')
        for word in _negative_words:
            if word in content:
                return '0'
        return '99'

    def extract_structural_info(self, content):
        structural_info = {}
        topic_range = self.find_topic_range(content, self.topic_pipeline.keys())
        for item in topic_range:
            begin, end, topic = item[0][0], item[0][1], item[1]
            topic_content = [(k, v) for k, v in self.word_segger.cut(content[begin:end], HMM=False)]
            structural_info[topic] = {}
            num = 1
            for template in self.topic_pipeline[topic]:
                for info in template(topic_content):
                    structural_info[topic][str(num)] = info
                    num += 1
        return structural_info

    def create_split_text(self, content):
        split_dat = {}
        split_dat['cut'] = '/'.join([ k.encode('utf8') for k, _ in self.word_segger.cut(content, HMM=False)])
        return split_dat


    def conduct_mri_structural_info(self, pipe_type='struct'):
        with open(self.output_path, 'w') as f_output:
            dat = {}
            num = 1
            for structural_info in self.execute_pipeline(pipe_type):
                dat[num] = structural_info
                num += 1
                # f_output.write(json.dumps(structural_info, indent=4, ensure_ascii=False, sort_keys=True) + CLRF)
            f_output.write(json.dumps(dat, indent=4, ensure_ascii=False, sort_keys=True))

    def create_train_and_validation_data(self, validation_ratio=0.3):
        # check validation ratio
        try:
            ratio = float(validation_ratio)
            if ratio > 1 or ratio < 0:
                raise ValueError('validation ratio \'%s\' not valid' % validation_ratio)
        except Exception, e:
            raise TypeError('validation ratio \'%s\' not float' % validation_ratio)
        # word encoding
        word_index = {}
        n_tokens = 0
        for k, v in sorted(xh_pipeline.word_frequency.items(), key=lambda x: x[1], reverse=True):
            if v > 5:
                word_index[k] = n_tokens
                n_tokens += 1
        NAN = n_tokens + 1
        # create label data
        max_sentence_length = 0
        data4lstm = []
        with open(self.label_data_path, 'w') as f_output:
            for content in self.read_mri_report():
                exam = content[0]
                result = content[1]
                a_piece_of_data = {}
                a_piece_of_data['label'] = 0
                if '考虑宫颈癌' in result:
                    a_piece_of_data['label'] = 1
                data = []
                for k, v in self.word_segger.cut(content[0], HMM=False):
                    if word_index.get(k, None):
                        data.append(int(word_index[k]))
                a_piece_of_data['data'] = data
                a_piece_of_data['data_length'] = len(data)
                if len(data) > max_sentence_length:
                    max_sentence_length = len(data)
                data4lstm.append(a_piece_of_data)
        # st(context=21)
        print max_sentence_length
        # shuffle all the data
        idx = range(len(data4lstm))
        np.random.shuffle(idx)
        # create data for train
        count_label1 = 0
        data4lstm_validation = []
        for i in idx[0:int(len(idx) * ratio)]:
            data4lstm_validation.append(data4lstm[i])
            count_label1 += data4lstm[i]['label']
        print("Validation data:")
        print("\tlabel 1 %s" % count_label1)
        print("\tlabel 0 %s" % (int(len(idx) * ratio) - count_label1))
        cPickle.dump(data4lstm_validation, open('./data/output/data4lstm_test', 'wb'),
                     protocol=cPickle.HIGHEST_PROTOCOL)
        # create data for validation
        count_label1 = 0
        data4lstm_train = []
        for i in idx[int(len(idx) * ratio):]:
            data4lstm_train.append(data4lstm[i])
            count_label1 += data4lstm[i]['label']
        print("Train data:")
        print("\tlabel 1 %s" % count_label1)
        print("\tlabel 0 %s" % (len(idx) - int(len(idx) * ratio) - count_label1))
        cPickle.dump(data4lstm_train, open('./data/output/data4lstm_train', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    xiehe_topic_pipeline = OrderedDict([
        # ('膀胱', []),
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

    # input_path = './data/input/MRI.csv'
    input_path = './data/input/all_mri.csv'
    output_path = './data/output/mri_describe_result.dat'

    label_data_path = './data/output/label_data.dat'

    word_segger = poseg

    xh_pipeline = XHPipeline(
        input_path, columns, output_path,
        usr_dict_path, usr_suggest_path,
        label_data_path,
        word_segger,
        xiehe_topic_pipeline
    )
    xh_pipeline.conduct_mri_structural_info(pipe_type='cut')

    with open('./data/output/word_frequency.dat', 'w') as f:
        for k, v in sorted(xh_pipeline.word_frequency.items(), key=lambda x: x[1]):
            if v > 5:
                f.write(k.encode('utf-8') + ',' + str(v) + CLRF)

    xh_pipeline.create_train_and_validation_data()

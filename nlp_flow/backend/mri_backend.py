# encoding=utf8

import jieba.posseg as poseg
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import sys

sys.path.append('../')
from utils.prepare_utils import JiebaTuning as JT
from utils.prepare_utils import GeneralAddressing as GA
import json
import cPickle
import numpy as np
import re

from ipdb import set_trace as st

JOIN = '/'
CLRF = '\n'
THRESHOLD_FREQ = 5


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

    SUB_AREA, ITEM, ASSERTION, DESC = '子区域', '指标', '断言', '描述'

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
                tmp_info[cls.SUB_AREA] = '_'.join(reversed(pre)).encode('utf-8')
                tmp_info[cls.ITEM] = tc[0].encode('utf-8')
                tmp_info[cls.ASSERTION] = neg_or_pos.encode('utf-8')
                tmp_info[cls.DESC] = '_'.join(des).encode('utf-8')
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
                tmp_info[cls.ITEM] = tc[0].encode('utf-8')
                tmp_info[cls.DESC] = (''.join(_tmp)).encode('utf-8')
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
                tmp_info[cls.ITEM] = tc[0].encode('utf-8')
                tmp_info[cls.ASSERTION] = has_or_not
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
                for i in range(index - 1, -1, -1):
                    if topic_content[i][0] in cls._deny_word:
                        if_deny = '无'
                    if topic_content[i][1] == 'x':
                        break
                for i in range(index + 1, len(topic_content)):
                    if topic_content[i][1] == 'x':
                        break
                    if topic_content[i][0] in cls._deny_word:
                        if topic_content[i][1] == 'd' and i < len(topic_content) - 1 and topic_content[i + 1][0] in (
                                u'均匀',):
                            continue
                        if_deny = '无'
                    if topic_content[i][0] == u'低信号':
                        high_or_low = '低信号'
                tmp_info = {}
                tmp_info[cls.ITEM] = tc[0].encode('utf-8')
                tmp_info[cls.ASSERTION] = if_deny
                tmp_info[cls.DESC] = high_or_low
                ret.append(tmp_info)
        return ret


class XHPipeline(object):
    _stop_words = [
        u',', u':', u';', u'.', u'', u'(', u')', u'×', u'"', u'-', u'*',
        u'并', u'的', u'于', u'者', u'在', u'另', u'与', u'一', u'其', u'有',
        u'约', u'呈', u'信号', u'略', u'或', u'所示', u'其它', u'等', u'稍',
        u'达', u'区', u'为', u'位', u'范围', u'扫描', u'段', u'显示', u'位于',
        u'直径约', u'序列', u'程度', u'性', u'扫', u'行', u'走', u'值', u'以',
        u'显', u'至', u'受', u'分别', u'局限', u'子', u'带', u'线', u'最',
        u'为主', u'缘', u'中', u'区域', u'数量', u'水平', u'灶', u'平', u'处',
        u'皆', u'样', u'及'
    ]

    _domain_knowledge_feature = ('DWI',)

    def __init__(self,
                 input_path, columns, output_path,
                 usr_dict_path, usr_suggest_path,
                 confirm_word_path,
                 hitting_word_path,
                 release_word_path,
                 label_data_path,
                 bow_data_path,
                 word_segger, topic_pipeline):
        self.input_path = input_path
        self.columns = columns
        self.output_path = output_path
        self.usr_dict_path = usr_dict_path
        self.usr_suggest_path = usr_suggest_path
        self.confirm_word_path = confirm_word_path
        self.hitting_word_path = hitting_word_path
        self.release_word_path = release_word_path
        self.label_data_path = label_data_path
        self.bow_data_path = bow_data_path
        self.word_segger = word_segger
        self.topic_pipeline = topic_pipeline
        self.word_frequency = defaultdict(int)
        self.hitting_and_releasing()

    def hitting_and_releasing(self):
        _confirm_words = []
        with open(self.confirm_word_path, 'r') as f:
            for l in f.xreadlines():
                _confirm_words.append(l.rstrip())
        _hitting_words = []
        with open(self.hitting_word_path, 'r') as f:
            for l in f.xreadlines():
                _hitting_words.append(l.rstrip())
        _release_words = []
        with open(self.release_word_path, 'r') as f:
            for l in f.xreadlines():
                _release_words.append(l.rstrip())
        self.confirm_words, self.hitting_words, self.release_words = \
            _confirm_words, _hitting_words, _release_words

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
            pos = -1
            topic_term = topic[0]
            for _term in topic:
                pos = content.find(_term)
                if pos >= 0:
                    topic_term = _term
                    break
            if pos >= 0:
                if content[pos + len(topic_term):pos + len(topic_term) + 1] not in _legal_punctuation:
                    _nearest_right_comma_pos = max([content[:pos].rfind(p) for p in _pos_punctuation]) + 1
                    ret.append([_nearest_right_comma_pos, topic_term])
                else:
                    ret.append([pos, topic_term])
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

    def get_word_frequency(self):
        for content in self.read_mri_report():
            # count word in mri exam desc
            for word, pos in self.word_segger.cut(content[0], HMM=False):
                if pos not in ('x',):
                    try:
                        int(word)
                    except Exception, e:
                        cond1 = 'mm' not in word
                        cond2 = 'cm' not in word
                        cond3 = word not in self._stop_words
                        # cond4 = word not in self._domain_knowledge_feature
                        if cond1 and cond2 and cond3:
                            self.word_frequency[word] += 1
                            # count word in mri result desc

    def execute_pipeline(self, pipe_type):
        for content in self.read_mri_report():
            if pipe_type == 'struct':
                yield self.extract_structural_info(content[0], content[1])

    def extract_structural_info(self, content, result):

        def _extract_domain_knowledge_feature(info):
            feature_weight = defaultdict(int)
            feature = info.get(XHLanguageTemplate.ITEM, None)
            if feature in self._domain_knowledge_feature:
                if feature == 'DWI':
                    cond1 = info.get(XHLanguageTemplate.DESC, None) == '高信号'
                    cond2 = info.get(XHLanguageTemplate.ASSERTION, None) == '有'
                    feature_weight['DWI'] += 1 if (cond1 and cond2) else 0
            return feature_weight

        structural_info = {}
        structural_info['content'] = content
        structural_info['result'] = result
        structural_info['feature_weight'] = []
        topic_range = self.find_topic_range(content, self.topic_pipeline.keys())
        for item in topic_range:
            begin, end, topic = item[0][0], item[0][1], item[1]
            topic_content = [(k, v) for k, v in self.word_segger.cut(content[begin:end], HMM=False)]
            structural_info[topic] = {}
            structural_info[topic]['range'] = {'begin': str(begin), 'end': str(end)}
            # structural_info[topic]['0'] = ' '.join(
            #     [tc[0].encode('utf-8') + '/' + tc[1].encode('utf-8') for tc in topic_content]
            # )
            num = 1
            _pipeline = []
            for cur_topics, cur_pipeline in self.topic_pipeline.items():
                if topic in cur_topics:
                    _pipeline = cur_pipeline
            for template in _pipeline:
                for info in template(topic_content):
                    structural_info[topic][str(num)] = info
                    domain_knowledge_feature = _extract_domain_knowledge_feature(info)
                    if len(domain_knowledge_feature.keys()) > 0:
                        structural_info['feature_weight'].append(domain_knowledge_feature)
                    num += 1
        return structural_info

    def conduct_mri_flow(self, pipe_type='struct'):
        with open(self.output_path, 'w') as f_output:
            dat = {}
            num = 1
            for structural_info in self.execute_pipeline(pipe_type):
                dat[num] = structural_info
                num += 1
            f_output.write(json.dumps(dat, indent=4, ensure_ascii=False, sort_keys=True))

    def determine_label_by_rule(self, exam, result):
        label = 0
        # confirm 1 rule
        for word in self.confirm_words:
            if word in result:
                label = 1
                return label
        # hitting rule
        for word in self.hitting_words:
            if word in result:
                label = 1
                break
        # releasing rule
        if label == 1:
            for word in self.release_words:
                if word in result:
                    label = 0
                    break
        return label

    def add_knowledge_driven_feature(self, exam, result):
        info = self.extract_structural_info(exam, result)

        _for_display = info['feature_weight']
        _for_pkl = np.zeros(len(self._domain_knowledge_feature))

        feature_index = {f: i for i, f in enumerate(self._domain_knowledge_feature)}
        for feature_weight in info['feature_weight']:
            for feature, weight in feature_weight.items():
                _for_pkl[feature_index[feature]] += weight
        return _for_display, _for_pkl

    def create_model_data(self, exam, result, word_index, n_tokens, model_type='bow', if_knowledge_driven=False):
        if model_type not in ('bow', 'lstm',):
            raise TypeError("model_type \"%s\" not valid " % model_type)
        label = self.determine_label_by_rule(exam, result)
        # for bow data
        # st(context=21)
        if model_type == 'bow':
            bow_data_display = {}
            bow_data_display['label'] = label
            bow_data_display['data'] = np.zeros(n_tokens + 1)
            bow_data_display['check'] = []
            bow_data_pkl = {}
            bow_data_pkl['label'] = label
            bow_data_pkl['data'] = np.zeros(n_tokens + 1)
            bow_data_pkl['check'] = []
            for k, v in self.word_segger.cut(exam, HMM=False):
                if word_index.get(k, None):
                    # for model in pkl format
                    bow_data_pkl['data'][int(word_index[k])] += 1
                    bow_data_pkl['check'].append('/'.join([k, str(word_index[k])]))
                    # for display in json format
                    bow_data_display['data'][int(word_index[k])] += 1
                    bow_data_display['check'].append('/'.join([k, str(word_index[k])]))

            if if_knowledge_driven:
                _feature_display, _feature_pkl = self.add_knowledge_driven_feature(exam, result)
                bow_data_display['domain_feature'] = _feature_display
                bow_data_pkl['data'] = np.concatenate((bow_data_pkl['data'], _feature_pkl), axis=0)

            bow_data_display['data'] = \
                '  '.join([str(i) + ':' + str(c) for i, c in enumerate(bow_data_display['data']) if c > 0]).encode(
                    'utf-8')
            bow_data_display['check'] = ' '.join(bow_data_display['check']).encode('utf-8')
            return bow_data_display, bow_data_pkl
        # for lstm data
        if model_type == 'lstm':
            feature_data = {}
            feature_data['label'] = label
            feature_data['data'] = []
            for k, v in self.word_segger.cut(exam, HMM=False):
                if word_index.get(k, None):
                    feature_data['data'].append(int(word_index[k]))
            feature_data['data_length'] = len(feature_data['data'])
            return feature_data

    def create_train_and_validation_data(self, threshold_freq=15, validation_ratio=0.3):
        # word encoding
        word_index = {}
        n_tokens = 0
        # filter word by frequency
        for k, v in sorted(self.word_frequency.items(), key=lambda x: x[1]):
            if v > threshold_freq:
                word_index[k] = n_tokens
                n_tokens += 1
        # create label data
        data4BOW_display = {}
        data4BOW_pkl = []
        num = 1
        with open(self.bow_data_path, 'w') as f_bow_output:
            for content in self.read_mri_report():
                exam = content[0]
                result = content[1]
                bow_data_display, bow_data_pkl = self.create_model_data(
                    exam,
                    result,
                    word_index,
                    n_tokens,
                    'bow',
                    # True
                )
                # st(context=21)
                data4BOW_display[num] = bow_data_display
                data4BOW_pkl.append(bow_data_pkl)
                num += 1
            f_bow_output.write(json.dumps(data4BOW_display, indent=4, ensure_ascii=False, sort_keys=True) + CLRF)
            cPickle.dump(data4BOW_pkl, open('../4bow/bow.pkl', 'wb'), protocol=cPickle.HIGHEST_PROTOCOL)
            # self.shuffle_and_store_data(data4BOW_pkl, './4bow/bow', 0.3)

    def shuffle_and_store_data(self, data, path, validation_ratio=0.3):
        # shuffle all the data
        idx = range(len(data))
        np.random.shuffle(idx)
        # create train / validation / test range point
        train_begin, train_end = 0, int(len(idx) * (1 - validation_ratio))
        valid_begin, valid_end = train_end, int(len(idx))
        # create data for train
        with open(path + '_train', 'wb') as f_train:
            count_label1 = 0
            data_train = []
            for i in idx[train_begin:train_end]:
                data_train.append(data[i])
                count_label1 += data[i]['label']
            print("Train data:")
            print("\tlabel 1 %s" % count_label1)
            print("\tlabel 0 %s" % (train_end - train_begin - count_label1))
            cPickle.dump(data_train, f_train, protocol=cPickle.HIGHEST_PROTOCOL)
        # create data for validation
        with open(path + '_valid', 'wb') as f_valid:
            count_label1 = 0
            data_valid = []
            for i in idx[valid_begin:valid_end]:
                data_valid.append(data[i])
                count_label1 += data[i]['label']
            print("Validation data:")
            print("\tlabel 1 %s" % count_label1)
            print("\tlabel 0 %s" % (valid_end - valid_begin - count_label1))
            cPickle.dump(data_valid, f_valid, protocol=cPickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    xiehe_topic_pipeline = OrderedDict([
        (('其它', '其他', '膀胱',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('宫体', '子宫'), [XHLanguageTemplate.one, XHLanguageTemplate.two, XHLanguageTemplate.four]),
        (('宫颈',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('阴道及外阴',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('双侧附件', '卵巢',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('淋巴结',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('直肠壁',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('髂骨',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('腹膜',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('臀大肌',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('直肠膀胱三角',), [XHLanguageTemplate.one, XHLanguageTemplate.four]),
        (('盆腔积液',), [XHLanguageTemplate.one, XHLanguageTemplate.three]),
    ])
    columns = ['ExamDescExResult', 'ResultDescExResult']

    usr_dict_path = '../data/input/mri_dict.dat'
    usr_suggest_path = '../data/input/mri_suggest.dat'

    # input_path = '../data/input/MRI.csv'
    input_path = '../data/input/all_mri.csv'
    output_path = '../data/output/mri_describe_result.dat'
    label_data_path = '../data/output/label_data.json'
    bow_data_path = '../data/output/bow_data.json'

    confirm_word_path = '../data/input/confirm_words.dat'
    hitting_word_path = '../data/input/hitting_words.dat'
    release_word_path = '../data/input/release_words.dat'

    word_segger = poseg

    xh_pipeline = XHPipeline(
        input_path, columns, output_path,
        usr_dict_path, usr_suggest_path,
        confirm_word_path,
        hitting_word_path,
        release_word_path,
        label_data_path,
        bow_data_path,
        word_segger,
        xiehe_topic_pipeline
    )

    # count word frequency
    xh_pipeline.get_word_frequency()
    with open('../data/output/word_frequency.pkl', 'wb') as f, \
            open('../data/output/word_frequency_display.csv', 'w') as f_display:
        word_frequency = {}
        for k, v in sorted(xh_pipeline.word_frequency.items(), key=lambda x: x[1]):
            if v > THRESHOLD_FREQ:
                word_frequency[k] = v
                f_display.write(k.encode('utf-8') + ',' + str(v) + CLRF)
        cPickle.dump(word_frequency, f)
    # conduct mri analyze flow
    xh_pipeline.conduct_mri_flow(pipe_type='struct')
    xh_pipeline.create_train_and_validation_data(THRESHOLD_FREQ, 0.3)

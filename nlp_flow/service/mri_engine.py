# encoding=utf8

from flask import Flask, request, jsonify
from collections import OrderedDict
import cPickle

import jieba.posseg as poseg
from xgboost import Booster, DMatrix

import sys

sys.path.append('../')
from backend.mri_backend import XHPipeline, XHLanguageTemplate
from backend.mri_backend import THRESHOLD_FREQ
from utils.prepare_utils import GeneralAddressing as GA

from ipdb import set_trace as st

app = Flask(__name__)


def create_pipeline_instance():
    topic_pipeline = OrderedDict([
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
    usr_dict_path = '../data/input/mri_dict.dat'
    usr_suggest_path = '../data/input/mri_suggest.dat'
    input_path = '../data/input/all_mri.csv'
    output_path = '../data/output/mri_describe_result.dat'
    label_data_path = '../data/output/label_data.json'
    bow_data_path = '../data/output/bow_data.json'
    confirm_word_path = '../data/input/confirm_words.dat'
    hitting_word_path = '../data/input/hitting_words.dat'
    release_word_path = '../data/input/release_words.dat'
    word_segger = poseg
    columns = ['ExamDescExResult', 'ResultDescExResult']
    xh_pipeline = XHPipeline(
        input_path, columns, output_path,
        usr_dict_path, usr_suggest_path,
        confirm_word_path,
        hitting_word_path,
        release_word_path,
        label_data_path,
        bow_data_path,
        word_segger,
        topic_pipeline
    )
    return xh_pipeline


def create_predictor_infos():
    word_index = {}
    n_tokens = 0
    with open('../data/output/word_frequency.pkl', 'rb') as f:
        word_frequency = cPickle.load(f)
        assert type(word_frequency) == dict
        for k, v in sorted(word_frequency.items(), key=lambda x: x[1]):
            if v > THRESHOLD_FREQ:
                word_index[k] = n_tokens
                n_tokens += 1
    bst = Booster()
    bst.load_model('../data/model/xgboost_model.model')
    return word_index, n_tokens, bst


g_pipeline_instance = create_pipeline_instance()
g_word_index, g_n_tokens, g_clf = create_predictor_infos()

_mask_items = ('content', 'result', 'feature_weight',)


@app.route('/extract-structural-info', methods=['POST', 'GET'])
def extract_structural_info():
    assert request.method == 'POST'
    data = request.form['mri_exam']
    data = GA.execute_general_addressing(
        data.encode('utf8'), [GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head]
    )
    info = g_pipeline_instance.extract_structural_info(data, '')
    for item in _mask_items:
        if info.has_key(item):
            del info[item]
    return jsonify(info)


@app.route('/mri-outcome-score', methods=['POST', 'GET'])
def model_feature_weight():
    assert request.method == 'POST'
    data = request.form['mri_exam']
    data = GA.execute_general_addressing(
        data.encode('utf8'), [GA.replace_punctuation, GA.negative_positive, GA.clear_trivial_head]
    )
    _, bow_data_pkl = g_pipeline_instance.create_model_data(data, '', g_word_index, g_n_tokens, 'bow')
    d_val = DMatrix(data=bow_data_pkl['data'].reshape(1,-1))
    prob = g_clf.predict(d_val)
    return str(prob[0])


@app.route('/', methods=['GET', 'POST'])
def default_page():
    return 'test for connection'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

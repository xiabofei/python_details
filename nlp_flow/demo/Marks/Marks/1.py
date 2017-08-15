#coding=utf8

from jinja2 import Template

import json

# f = open('../static/data/mri_result.json')
f = open('mri_result.json')
data = json.load(f)

text = ""  # 原文
areas = dict()  # 每个区域为一组(宫体,宫颈...)
gongti_pos = []  # 每个指标的位置,数组
gongjing_pos = []

text = data[u'MRI原文']
all_areas = data[u'MRI结构化']

for k,v in all_areas.iteritems():
    for k1,v1 in v[u'抽取结果'].iteritems():
        splitArea = v1[u'区域'].split('_')
        all_areas[k][u'抽取结果'][k1][u'区域']=splitArea
        splitDesc = v1[u'描述'].split('_')
        all_areas[k][u'抽取结果'][k1][u'描述']=splitDesc

for k,v in all_areas.items():
    print k
    for k1,v1 in v.iteritems():
        if isinstance(v1,dict):
            for k2,v2 in v1.iteritems():
                for k3,v3 in v2.iteritems():
                    print k3,v3





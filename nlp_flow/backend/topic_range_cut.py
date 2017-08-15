# encoding=utf8

topics = [
    ('其它', '其他', '膀胱',),
    ('宫体', '子宫'),
    ('宫颈',),
    ('阴道及外阴',),
    ('双侧附件', '卵巢',),
    ('淋巴结',),
    ('直肠壁',),
    ('髂骨',),
    ('腹膜',),
    ('臀大肌',),
    ('直肠膀胱三角',)
]

content = '子宫呈前倾位,子宫内膜长T2信号显示欠清,子宫肌层信号不均,可见多发短T2信号,宫颈未见增大,信号欠均匀,增强呈轻度强化,膀胱充盈欠佳.左侧附件区见长T2信号影,直径约为15mm,右侧附件区未见明显异常信号.直肠壁未见明显增厚.双侧髂骨骨质信号无异常,双侧臀大肌及髂腰肌信号无异常改变.直肠膀胱三角内未见异常信号影.盆腔及腹股沟区未见异常肿大淋巴结'


def find_topic_range(content, topics):
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


res = find_topic_range(content, topics)
print res

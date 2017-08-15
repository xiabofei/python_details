# encoding=utf8

import requests
from ipdb import set_trace as st
import json
import collections

def convert(data):
    if isinstance(data, basestring):
        return data.encode('utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.iteritems()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data.encode('utf-8')

def structural_test():
    r = requests.post('http://127.0.0.1:5002/extract-structural-info', data=data)
    area_info = r.json()
    json.dump(convert(area_info), open('./post_res.json', 'w'), indent=4, sort_keys=True, ensure_ascii=False)
    print(r.text)

def classifier_test():
    r = requests.post('http://127.0.0.1:5002/mri-outcome-score', data=data)
    print r.text

if __name__ == '__main__':
    data = {
        "mri_exam": "膀胱充盈可,形态正常,壁光整,未见明显增厚.子宫颈缩小,T2WI信号减低,宫颈右后壁见小片状稍长T2信号,增强后强化减低,子宫腔扩大,前后径约16.8mm,可见长T2信号,结合带完整.双侧附件区未见明显异常信号.直肠壁未见明显增厚.双侧髂骨骨质信号无异常,双侧臀大肌及髂腰肌信号无异常改变.直肠膀胱三角内未见异常信号影.盆腔双侧髂血管旁见多个小淋巴结.",
        # "mri_exam": "宫体:子宫呈前倾位,宫内膜显示欠清,宫腔增宽,积液,呈短T1长T2信号改变.结合带显示欠清.宫颈:弥漫性不规则增厚,累及长度约6cm,向上侵犯宫体下部,向下侵犯阴道上壁,结合带,内膜及浆膜层显示欠清,病变突破浆膜层侵犯宫旁脂肪间隙,其内可见多发尖角样突起,毛刺,病变局部与直肠前壁分界欠清.阴道及外阴:阴道上壁与宫颈病变分界不清,前,后穹窿显示欠清.双侧附件:附件未见明显占位;淋巴结:宫颈左侧可见少许小淋巴结,短径约5mm.其它:髂血管,膀胱,输尿管等盆腔脏器未见明显侵犯;盆腔积液:未见明确显示;骨盆骨质:未见明确骨转移."
    }
    structural_test()
    # classifier_test()

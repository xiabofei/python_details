#encoding=utf8

from collections import defaultdict
from collections import OrderedDict
from ipdb import set_trace as st
import json
import yaml

SPLIT_AMONG_FIELD = '\t'
CRLF = '\n'
SPLIT_INNER_FIELD = '+'

ADM_EMR_LIS_PATH = './extracted_data/merged_adm_lis_emr.dat'
OUTPUT_PATH = './extracted_data/result.dat'

def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

if __name__ == '__main__':
    # 0954532+7096589 20140805 {"LIS": {"6966003||4+肾全": {"天门冬氨酸氨基转移酶": "16", "肌>      酐(酶法)": "63", "白蛋白": "42", "无机磷": "1.36", "直接胆红素": "2.0", "谷氨酰转肽酶":       "26", "丙氨酸氨基转移酶": "14", "尿酸": "394", "氯": "100", "胆碱酯酶": "8.9", "尿素": "      4.25", "钾": "4.3", "总胆汁酸": "2.4", "葡萄糖": "12.9", "乳酸脱氢酶": "199", "前白蛋白"      : "239", "钠": "136", "碱性磷酸酶": "87", "钙": "2.32", "总二氧化碳": "26.6", "总胆红素"      : "8.5", "总蛋白": "71", "白蛋白球蛋白比": "1.4"}, "6966003||3+肾全": {"天门冬氨酸氨基转      移酶": "16", "肌酐(酶法)": "63", "白蛋白": "42", "无机磷": "1.36", "直接胆红素": "2.0",       "谷氨酰转肽酶": "26", "丙氨酸氨基转移酶": "14", "尿酸": "394", "氯": "100", "胆碱酯酶":       "8.9", "尿素": "4.25", "钾": "4.3", "总胆汁酸": "2.4", "葡萄糖": "12.9", "乳酸脱氢酶": "      199", "前白蛋白": "239", "钠": "136", "碱性磷酸酶": "87", "钙": "2.32", "总二氧化碳": "2      6.6", "总胆红素": "8.5", "总蛋白": "71", "白蛋白球蛋白比": "1.4"}, "6966003||5+全血细胞>      分析": {"中性粒细胞绝对值": "3.11", "血红蛋白": "127", "嗜酸性粒细胞百分比": "6.1", "嗜>      酸性粒细胞绝对值": "0.35", "单核细胞百分比": "6.1", "红细胞体积分布宽度(CV)": "12.4", ">      淋巴细胞百分比": "32.8", "白细胞": "5.74", "嗜碱性粒细胞百分比": "0.9", "平均红细胞血红>      蛋白浓度": "332", "单核细胞绝对值": "0.35", "红细胞": "4.22", "中性粒细胞百分比": "54.1"      , "红细胞体积分布宽度(SD)": "40.8", "大血小板比率": "19.3", "平均血小板体积": "9.4", "血      小板体积分布宽度": "10.4", "嗜碱性粒细胞绝对值": "0.05", "平均红细胞血红蛋白": "30.1", "      血小板压积": "0.30", "淋巴细胞绝对值": "1.88", "平均红细胞体积": "90.8", "红细胞压积": "      38.3", "血小板": "324"}, "6966003||8+鳞状细胞癌抗原(SCCAg)": {"鳞状细胞癌抗原": "0.5"}},       "EMR": {}}

    with open(ADM_EMR_LIS_PATH, 'r') as f_adm_emr_lis, open(OUTPUT_PATH, 'w') as f_output:

        last_patientID = ""
        last_date = ""
        line_content_to_write = OrderedDict([('patientID',None), ('info',defaultdict(dict))])

        for l in f_adm_emr_lis.xreadlines():
            try:

                fields = l.split(SPLIT_AMONG_FIELD)
                patientID, caseID = fields[0].split(SPLIT_INNER_FIELD)[0], fields[0].split(SPLIT_INNER_FIELD)[1]
                date = fields[1].strip()
                content = {}
                if len(fields)>2:
                    content = json_loads_byteified(fields[2])

                if last_patientID==patientID:
                    line_content_to_write['info'][str(date)][str(caseID)] = content
                else:
                    if last_patientID!="":
                        f_output.write(json.dumps(line_content_to_write, ensure_ascii=False)+CRLF)
                    last_patientID = patientID
                    last_date = date
                    line_content_to_write = OrderedDict([('patientID',None), ('info',defaultdict(dict))])
                    line_content_to_write['patientID'] = str(patientID)
                line_content_to_write['info'][str(date)][str(caseID)] = content

            except Exception,e:
                st(context=21)
                print e

        f_output.write(json.dumps(line_content_to_write, ensure_ascii=False)+CRLF)


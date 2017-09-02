# coding=utf8

import os
import re
import sys
import functools

from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf-8')

CLRF = '\n'

from ipdb import set_trace as st


class EMRAnalyzer(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.patient_id = defaultdict(int)

    def emr_data_processor(self, method, f_name_filter):
        curr_path = []
        curr_path.append(self.dir_name)
        for month in filter(self._filter_dir, os.listdir(self.dir_name)):
            curr_path.append(month)
            for day in filter(self._filter_dir, os.listdir('/'.join(curr_path))):
                curr_path.append(day)
                for f_name in sorted(filter(f_name_filter, os.listdir('/'.join(curr_path)))):
                    if method.im_func.func_name in ('content_iterator',):
                        yield method(curr_path, f_name)
                    else:
                        method(curr_path, f_name)
                curr_path.pop()
            curr_path.pop()

    def content_iterator(self, curr_path, f_name):
        content = open(os.path.join('/'.join(curr_path), f_name)).read()
        return (content, self.name_to_id(f_name))

    def patient_id_counter(self, curr_path, f_name):
        self.patient_id[self.name_to_id(f_name)] += 1

    @classmethod
    def no_filter(cls, f_name):
        return True

    @classmethod
    def name_to_id(cls, f_name):
        return f_name.split('_')[1]

    @classmethod
    def _filter_dir(cls, dir_name):
        return '.' != dir_name[0]

    @classmethod
    def _filter_name(cls, f_name):
        return '住院志' in f_name

    @classmethod
    def _filter_name_qita(cls, f_name):
        return '其他记录' in f_name

    @classmethod
    def _filter_name_bingcheng(cls, f_name):
        return '病程记录' in f_name

    def ruyuan_file(cls, f_name):
        return f_name in cls.f_names

    @classmethod
    def _clean_content(cls, content):
        return content.replace('　', '').strip()

    @classmethod
    def extract_elem_name_jiwangshi(cls, content):
        content = re.sub("<section name=\"既往史\" code=\"\" code-system=\"\"", "既往史", content)
        return content

    @classmethod
    def extract_elem_name_keshi(cls, content):
        content = re.sub("<fieldelem name=\"科室\" code=\"\" code-system=\"\">", "科室", content)
        return content

    @classmethod
    def extract_content_from_one_record(cls, content):
        content = re.sub("code-system=\"\"", ">", content)
        pattern = re.compile(">{1,2}(.*?)<")
        return filter(lambda x: x, map(cls._clean_content, re.findall(pattern, content)))

    @classmethod
    def extract_ZHENDUAN(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '初步诊断' in c or '出院诊断' in c:
                begin = i
            if begin > 0 and ('医师' in c or '出院医嘱' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_ZHUSU(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '主诉' in c or '主  诉' in c or "主   诉" in c:
                begin = i
            if begin > 0 and ('现病史' in c or '既往史' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_XIANBINGSHI(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '现病史' in c:
                begin = i
            if begin > 0 and ('既往史' in c or '入院诊断' in c or '个人史' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_JIWANGSHI(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '既往史' in c:
                begin = i
            if begin > 0 and ('个人史' in c or '入院诊断' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_GERENSHI(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '个人史' in c:
                begin = i
            if begin > 0 and ('婚育史' in c or '婚姻史' in c or '家族史' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_BINGCHENGJILU(cls, content):
        begin, end = 0, 0
        for i, c in enumerate(content):
            if '首次病程记录' in c:
                begin = i
            if begin > 0 and ('副主任医师' in c or '记录者' in c or '主任医师' in c or '诊疗计划' in c or '住院医师' in c):
                end = i
                break
        return ''.join(content[begin + 1:end])

    @classmethod
    def extract_KESHI(cls, content):
        keshi = cls._extract_KESHI(content)
        keshi = re.sub('一|二|三|四|五|六|七|八|九|十|科|病区|住院|护士站|东院|康馨', '', keshi)
        keshi = keshi.replace('：', '').replace('、','')
        return keshi.strip()

    @classmethod
    def _extract_KESHI(cls, content):
        begin, end = 0, 0
        if '患者入院病情评估表' in content or '出院证' in content or '出 院 证' in content:
            for i, c in enumerate(content):
                if '科室' in c:
                    begin = i
                    if '床号' in c:
                        end = i
                        break
                if begin > 0 and ('床号' in c or '医师' in c):
                    end = i
                    break
            if begin == end and begin > 0:  # 科室:住院核医学科  床号:
                # 把标点转成str
                return content[begin].replace("：", "").split("科室")[1].split("床号")[0]
            elif begin + 1 == end:  # 科室:住院核医学科</text><text>床号:
                return content[begin].replace(":", "").split("科室")[1]
        return ''.join(content[begin + 1:end])






def write_to_local(f, f_name, content, code4sort):
    f.write('\t'.join([f_name, str(code4sort), unicode(content.replace('\t', '')).encode('utf-8')]) + CLRF)


def extract_from_zhuyuanzhi(dir_name, emr_analyzer):
    # extract data from '住院志'
    print('extract from zhuyuanzhi')
    with open('../data/output/1_extracted_zhenduan.dat', 'w') as f_zhenduan, \
            open('../data/output/2_extracted_zhusu.dat', 'w') as f_zhusu, \
            open('../data/output/3_extracted_xianbingshi.dat', 'w') as f_xianbingshi, \
            open('../data/output/4_extracted_jiwangshi.dat', 'w') as f_jiwangshi, \
            open('../data/output/5_extracted_gerenshi.dat', 'w') as f_gerenshi:
        nan_zhenduan, nan_zhusu, nan_xianbingshi, nan_jiwangshi, nan_gerenshi = 0, 0, 0, 0, 0
        for it in emr_analyzer.emr_data_processor(emr_analyzer.content_iterator, emr_analyzer._filter_name):
            content, f_name = it[0], it[1]
            content = emr_analyzer.extract_content_from_one_record(content)
            if content != '':
                # zhenduan
                content_zhenduan = emr_analyzer.extract_ZHENDUAN(content)
                nan_zhenduan += 1 if content_zhenduan == '' else 0
                write_to_local(f_zhenduan, f_name, content_zhenduan, 1)
                # zhusu
                content_zhusu = emr_analyzer.extract_ZHUSU(content)
                nan_zhusu += 1 if content_zhusu == '' else 0
                write_to_local(f_zhusu, f_name, content_zhusu, 2)
                # xianbingshi
                content_xianbingshi = emr_analyzer.extract_XIANBINGSHI(content)
                nan_xianbingshi += 1 if content_xianbingshi == '' else 0
                write_to_local(f_xianbingshi, f_name, content_xianbingshi, 3)
                # jiwangshi
                content_jiwangshi = emr_analyzer.extract_JIWANGSHI(content)
                nan_jiwangshi += 1 if content_jiwangshi == '' else 0
                write_to_local(f_jiwangshi, f_name, content_jiwangshi, 4)
                # gerenshi
                content_gerenshi = emr_analyzer.extract_GERENSHI(content)
                nan_gerenshi += 1 if content_gerenshi == '' else 0
                write_to_local(f_gerenshi, f_name, content_gerenshi, 5)


def extract_from_qitawenjian(dir_name, emr_analyzer):
    # extract from '其他文件'
    print('extract from qitawenjian')
    with open('../data/output/0_extracted_keshi.dat', 'w') as f_keshi:
        nan_count = 0
        cur_id = ''
        for it in emr_analyzer.emr_data_processor(emr_analyzer.content_iterator, emr_analyzer._filter_name_qita):
            content, f_name = it[0], it[1]
            if f_name != cur_id:  # 同一个id有很多其他记录,找到一个含有科室的即可
                content = emr_analyzer.extract_elem_name_keshi(content)
                content = emr_analyzer.extract_content_from_one_record(content)
                content_keshi = emr_analyzer.extract_KESHI(content)
                nan_count += 1 if content_keshi == '' else 0
                if content_keshi != '':
                    cur_id = f_name
                    write_to_local(f_keshi, f_name, content_keshi, 0)
        print nan_count


def extract_from_bingchengjilu(dir_name, emr_analyzer):
    # extract from '病程记录'
    print('extract from bingchengjilu')
    with open('../data/output/6_extracted_bingchengjilu.dat', 'w') as f_bingchengjilu:
        nan_count = 0
        for it in emr_analyzer.emr_data_processor(emr_analyzer.content_iterator, emr_analyzer._filter_name_bingcheng):
            content, f_name = it[0], it[1]
            content = emr_analyzer.extract_content_from_one_record(content)
            if content != '':
                content_bingchengjilu = emr_analyzer.extract_BINGCHENGJILU(content)
                nan_count += 1 if content_bingchengjilu=='' else 0
                write_to_local(f_bingchengjilu, f_name, content_bingchengjilu, 6)
        print nan_count


if __name__ == '__main__':
    dir_name = '/Users/xiabofei/Documents/emr-record/2015'
    emr_analyzer = EMRAnalyzer(dir_name)
    # extract_from_qitawenjian(dir_name, emr_analyzer)
    # extract_from_zhuyuanzhi(dir_name, emr_analyzer)
    # extract_from_bingchengjilu(dir_name, emr_analyzer)

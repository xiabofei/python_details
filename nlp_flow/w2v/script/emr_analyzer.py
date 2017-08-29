# coding=utf8

import os
import re
import sys

from collections import defaultdict

reload(sys)
sys.setdefaultencoding('utf-8')

CLRF = '\n'


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
    def _clean_content(cls, content):
        return content.replace('　', '').strip()

    @classmethod
    def extract_content_from_one_record(cls, content):
        content = re.sub("<section name=\"既往史\" code=\"\" code-system=\"\"", "既往史", content)
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


def write_to_local(f, f_name, content):
    f.write(f_name + '\t' + unicode(content).encode('utf-8') + CLRF)

if __name__ == '__main__':

    dir_name = '/Users/xiabofei/Documents/emr-record/2015'
    nan_count = 0
    emr_analyzer = EMRAnalyzer(dir_name)

    with open('../data/output/extracted_zhenduan.dat', 'w') as f_zhenduan, \
            open('../data/output/extracted_zhusu.dat', 'w') as f_zhusu, \
            open('../data/output/extracted_xianbingshi.dat', 'w') as f_xianbingshi, \
            open('../data/output/extracted_jiwangshi.dat', 'w') as f_jiwangshi, \
            open('../data/output/extracted_gerenshi.dat', 'w') as f_gerenshi:
        for it in emr_analyzer.emr_data_processor(emr_analyzer.content_iterator, emr_analyzer._filter_name):
            content, f_name = it[0], it[1]
            content = emr_analyzer.extract_content_from_one_record(content)
            if content == '':
                nan_count += 1
            else:
                # zhenduan
                content_zhenduan = emr_analyzer.extract_ZHENDUAN(content)
                write_to_local(f_zhenduan, f_name, content_zhenduan)
                # zhusu
                content_zhusu = emr_analyzer.extract_ZHUSU(content)
                write_to_local(f_zhusu, f_name, content_zhusu)
                # xianbingshi
                content_xianbingshi = emr_analyzer.extract_XIANBINGSHI(content)
                write_to_local(f_xianbingshi, f_name, content_xianbingshi)
                # jiwangshi
                content_jiwangshi = emr_analyzer.extract_JIWANGSHI(content)
                write_to_local(f_jiwangshi, f_name, content_jiwangshi)
                # gerenshi
                content_gerenshi = emr_analyzer.extract_GERENSHI(content)
                write_to_local(f_gerenshi, f_name, content_gerenshi)
        print nan_count

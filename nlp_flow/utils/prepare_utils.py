# encoding=utf8

import jieba

class JiebaTuning(object):

    @staticmethod
    def add_usr_dict(path):
        with open(path, 'r') as f:
            for l in f.xreadlines():
                items = l.split(',')
                jieba.add_word(items[0].rstrip(), items[1].rstrip(), items[2].rstrip())

    @staticmethod
    def suggest_usr_dict(path):
        with open(path, 'r') as f:
            for l in f.xreadlines():
                word1, word2 = l.split(',')[0].rstrip(), l.split(',')[1].rstrip()
                jieba.suggest_freq((word1, word2), True)


class GeneralAddressing(object):

    @staticmethod
    def replace_punctuation(content):
        _chinese_english = [
            ('，', ','),
            ('、', ','),
            ('（', '('),
            ('）', ')'),
            ('。', '.'),
            ('；', ';'),
            ('：', ':'),
            ('“', '"'),
            ('－', '-'),
            (' ', ''),
            ('Ca', '癌'),
            ("”", '"')
        ]
        for i in _chinese_english:
            content = content.replace(i[0], i[1])
        return content

    @staticmethod
    def clear_trivial_head(content):
        pos = content.find('复查')
        if pos >= 0:
            return content[pos + len('复查'):]
        else:
            return content

    @staticmethod
    def negative_positive(content):
        _symbol_word = [
            ('(-)', '阴性'),
            ('(+)', '阳性'),
        ]
        for i in _symbol_word:
            content = content.replace(i[0], i[1])
        return content

    @staticmethod
    def execute_general_addressing(content, process_flow):
        ret = content
        for flow in process_flow:
            ret = flow(ret)
        return ret


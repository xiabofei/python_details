#encoding=utf8

from ipdb import set_trace as st
from pprint import pprint

# # 中文分词(根据词库和算法)
# from nltk.tokenize import StanfordSegmenter
# segmenter = StanfordSegmenter(
#     path_to_sihan_corpora_dict='/Users/xiabofei/nltk_data/stanford-segmenter/data',
#     path_to_model='/Users/xiabofei/nltk_data/stanford-segmenter/data/pku.gz',
#     path_to_dict='/Users/xiabofei/nltk_data/stanford-segmenter/data/dict-chris6.ser.gz'
# )
# res= segmenter.segment(u"北海已经成为中国对外开放中升起的一颗明星")
# print(type(res))
# print(res.encode('utf-8'))
#
# # 英文分词(token化)
# from nltk.tokenize import StanfordTokenizer
# tokenizer = StanfordTokenizer()
# sent = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks."
# print tokenizer.tokenize(sent)
#
# # 英文命名实体识别
# from nltk.tag import StanfordNERTagger
# eng_tagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
# print eng_tagger.tag('Rami Eid is studying at Stony Brook University in NY'.split())
#
# # 中文命名实体识别
# chi_tagger = StanfordNERTagger('chinese.misc.distsim.crf.ser.gz')
# sent = u'北海 已 成为 中国 对外开放 中 升起 的 一 颗 明星'
# for word, tag in chi_tagger.tag(sent.split()):
#     print word.encode('utf-8'), tag
#
# # 英文词性标注
from nltk.tag import StanfordPOSTagger
# eng_tagger = StanfordPOSTagger('english-bidirectional-distsim.tagger')
# print eng_tagger.tag('What is the airspeed of an unladen swallow ?'.split())
# # 中文词性标注
chi_tagger = StanfordPOSTagger('chinese-distsim.tagger')
# sent = u'北海 已 成为 中国 对外开放 中 升起 的 一 颗 明星'
sent = u'宫体 子宫 呈 垂直位 宫内膜 高 T2 信号 连续'
for _, word_and_tag in chi_tagger.tag(sent.split()):
    word, tag = word_and_tag.split('#')
    print word.encode('utf-8'), tag


# 中英文句法分析 区别在于词库不同
from nltk.parse.stanford import StanfordParser
eng_parser = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz')
sent = list(u'子宫 呈 垂直位 , 宫内膜 高 T2 信号 连续'.split())
for tree in eng_parser.parse(sent):
    tree.pprint()


# 依存关系分析
from nltk.parse.stanford import StanfordDependencyParser
eng_parser = StanfordDependencyParser(model_path='edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz')
res = list(eng_parser.parse(u'子宫 呈 垂直位 , 宫内膜 高 T2 信号 连续'.split()))
# st(context=21)
for row in res[0].triples():
    print '(' + row[0][0] + ',' + row[0][1] + ')', row[1], '(' + row[2][0] + ',' + row[2][1] + ')'
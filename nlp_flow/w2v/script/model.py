# encoding=utf8


import gensim
import logging

from sentence_iterator import SentencesIterator


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# user define dict path
usr_dict_dir = '../data/input/domain_dict/'
usr_dict_files = (
    '0_central', '1_auxiliary', '2_manual',
    '3_zhenduan_addressed', '4_shoushu_addressed', '5_jianyan_addressed', '6_yaopin_addressed',
)
usr_dict_path_list = [usr_dict_dir + f for f in usr_dict_files]
# user suggest dict path
usr_suggest_path_list = []
# target keshi path name
target_keshi_path = '../data/input/target_keshi.dat'
# cut columns
cut_columns = ('zhenduan', 'zhusu', 'xianbingshi', )
# wanted column
wanted_column =  'xianbingshi'

assert wanted_column in cut_columns, "wanted column not in cut columns"


sentence_it = SentencesIterator(
    '../data/output/merged/all_merged.csv',
    usr_dict_path_list,
    usr_suggest_path_list,
    target_keshi_path,
    cut_columns,
    wanted_column,
    '\t',
    True
)


model = gensim.models.Word2Vec(sentence_it, min_count=30, size=32, workers=4, window=2)
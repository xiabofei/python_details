# encoding=utf8

import gensim
import logging
from collections import Counter
from collections import defaultdict
import operator
import re

from ipdb import set_trace as st

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

CLRF = '\n'


class RecommendCategory(object):
    # nan_category = 'nan'
    nan_category = ''

    # not_in_vocab = 'not in vocab'
    not_in_vocab = ''

    # too_less_hit = 'too less hit'
    too_less_hit = ''

    suffix_words = u'病|症|后|型|期|史|程|级|性|区|周|天$'

    def __init__(self, model_path, word_category_path):
        self.model_name = model_path.replace('.model', '').split('/')[-1]
        self.model = gensim.models.Word2Vec.load(model_path)
        self.word_category = {w: c for w, c in self.load_word_category(word_category_path)}

        # indicate the way by which the term is hit
        self.term_in_word_category = 'in'
        self.term_in_word_category_count = 0
        self.term_in_word_category_record = defaultdict(int)

        self.term_minus_suffix_in_word_category = 'suffix_in'
        self.term_minus_suffix_in_word_category_count = 0
        self.term_minus_suffix_in_word_category_record = defaultdict(int)

        self.term_not_in_vocab = 'not_in_vocab'
        self.term_not_in_vocab_count = 0
        self.term_not_in_vocab_record = defaultdict(int)

        self.term_similar_miss = 'miss_similar'
        self.term_similar_miss_count = 0
        self.term_similar_miss_record = defaultdict(int)

        self.term_similar = 'hit_similar'
        self.term_similar_count = 0
        self.term_similar_record = defaultdict(int)

    def reset_term_hit_type_counter(self):
        self.term_in_word_category_count = 0
        self.term_minus_suffix_in_word_category_count = 0
        self.term_not_in_vocab_count = 0
        self.term_similar_miss_count = 0
        self.term_similar_count = 0

    def display_term_hit_result(self):
        print('term_in_word_category_count:%s' % self.term_in_word_category_count)
        print('term_minus_suffix_in_word_category_count:%s' % self.term_minus_suffix_in_word_category_count)
        print('term_not_in_vocab_count:%s' % self.term_not_in_vocab_count)
        print('term_similar_miss_count:%s' % self.term_similar_miss_count)
        print('term_similar_count:%s' % self.term_similar_count)

    def write_term_hitting_record(self):
        def _write(records, path):
            with open(path, 'w') as f:
                f.write(
                    CLRF.join(
                        [
                            '\t'.join([r[0].encode('utf8'), str(r[1])])
                            for r in sorted(records.items(), key=lambda x: x[1], reverse=True)
                            ]
                    )
                )

        for i, records in enumerate([
            self.term_in_word_category_record,
            self.term_minus_suffix_in_word_category_record,
            self.term_not_in_vocab_record,
            self.term_similar_miss_record,
            self.term_similar_record]):
            _write(records, ('../data/output/cut_result/' + self.model_name + str(i)))

    def load_word_category(self, word_category_path):
        with open(word_category_path, 'r') as f:
            for l in f.readlines():
                yield (l.split('\t')[0].decode('utf8'), l.split('\t')[1].strip().decode('utf8'))

    def recommend_category(self, word, topn=10, mini_sim_ratio=1.0):
        # word in category
        if self.word_category.get(word, None):
            self.term_in_word_category_count += 1
            self.term_in_word_category_record[word] += 1
            return self.word_category.get(word), self.term_in_word_category
        # word-suffix in category
        if self.word_category.get(re.sub(self.suffix_words, "", word), None):
            self.term_minus_suffix_in_word_category_count += 1
            self.term_minus_suffix_in_word_category_record[word] += 1
            return self.word_category.get(re.sub(self.suffix_words, "", word)), self.term_minus_suffix_in_word_category
        # word and word-suffix not in w2v vocab
        if not self.model.wv.vocab.has_key(word):
            if not self.model.wv.vocab.has_key(re.sub(self.suffix_words, "", word)):
                self.term_not_in_vocab_count += 1
                self.term_not_in_vocab_record[word] += 1
                return self.not_in_vocab, self.term_not_in_vocab
            else:
                word = re.sub(self.suffix_words, "", word)
        # word or word-suffix in w2v vocab
        similar_words = self.model.most_similar(word, topn=topn)
        # logging.info('......topn similar word......')
        # for word_sim in similar_words:
        #     print word_sim[0], word_sim[1], self.word_category.get(word_sim[0], self.nan_category)
        category_weights = defaultdict(int)
        category_words = defaultdict(list)
        for i in similar_words:
            if self.word_category.get(i[0], None):
                category_weights[self.word_category.get(i[0])] += i[1]
                category_words[self.word_category.get(i[0])].append(i[0])
        if len(category_weights) > 0:
            # logging.info('......category count......')
            # word, count = Counter(candidate_categories).most_common(1)[0][0],Counter(candidate_categories).most_common(1)[0][1]
            self.term_similar_count += 1
            c, sim = max(category_weights.iteritems(), key=operator.itemgetter(1))
            self.term_similar_record['#'.join([word, '['+c+']', '|'.join(category_words[c])])] += 1
            return c, '|'.join(category_words[c])
        self.term_similar_miss_count += 1
        self.term_similar_miss_record[word] += 1
        return self.nan_category, self.term_similar_miss


def recommend_by_model(model_dir, model_name):
    word_category_path = '../data/output/domain_dict/snowball_category.csv'
    rc = RecommendCategory(model_dir + model_name, word_category_path)
    rc.reset_term_hit_type_counter()
    dis_cut_origin_path = '../data/output/cut_result/dis_cut.csv'
    dis_cut_recommend_path = '../data/output/cut_result/dis_cut_' + model_name.replace('.model', '') + '.csv'
    all_count, hit_count = 0, 0
    with open(dis_cut_origin_path, 'r') as f_in, open(dis_cut_recommend_path, 'w') as f_out:
        for l in f_in.readlines():
            items = l.strip().split('/')
            all_count += len(items)
            for index, content in enumerate(items):
                if ('[' not in content) and (']' not in content):
                    category, extra_info = rc.recommend_category(content.decode('utf8'))
                    if category:
                        items[index] += '[' + category.encode('utf8') + ']'
                        hit_count += 1
                    items[index] += '(' + extra_info.encode('utf8') + ')'
                else:
                    hit_count += 1
            f_out.write('/'.join(items) + CLRF)
    logging.info(model_name)
    print('hit / all = %s / %s = %s' % (hit_count, all_count, hit_count * 1.0 / all_count))
    rc.display_term_hit_result()
    rc.write_term_hitting_record()


if __name__ == '__main__':
    model_name = (
        # 'mincount10_size64_window2_small.model',
        # 'mincount10_size64_window2.model',
        'mincount5_size64_window2_dict_updated.model',
        'mincount5_size64_window5_dict_updated.model',
    )
    for mp in model_name:
        recommend_by_model('../data/input/', mp)

# encoding=utf8

import gensim
from collections import defaultdict
import operator
import re
from utils import CLRF
from utils import SEG_SPLIT
import json

from ipdb import set_trace as st

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Sug4Category(object):
    suffix_words = u'症$|后$|型$|期$|史$|程$|级$|性$|区$|周$|天$'

    def __init__(self, model_path, word_category_path):
        self.model_name = model_path.replace('.model', '').split('/')[-1]
        self.model = gensim.models.Word2Vec.load(model_path)
        self.word_category = {w: c for w, c in self.load_word_category(word_category_path)}
        self.word_cache = {}
        self.init_term_hit_type()

    def init_term_hit_type(self):
        self.in_word_category_info = 'in'
        self.in_word_category_counter = defaultdict(int)
        self.minus_suffix_in_word_category_info = 'suffix_in'
        self.minus_suffix_in_word_category_counter = defaultdict(int)
        self.similar_miss_info = 'miss_similar'
        self.similar_miss_counter = defaultdict(int)
        self.similar_info = 'hit_similar'
        self.similar_counter = defaultdict(int)
        self.not_in_vocab_info = 'not_in_vocab'
        self.not_in_vocab_counter = defaultdict(int)

    def display_term_hit_result(self):
        logging.info('term_in_word_category_count:%s' % sum(self.in_word_category_counter.values()))
        logging.info(
            'term_minus_suffix_in_word_category_count:%s' % sum(self.minus_suffix_in_word_category_counter.values()))
        logging.info('term_not_in_vocab_count:%s' % sum(self.not_in_vocab_counter.values()))
        logging.info('term_similar_miss_count:%s' % sum(self.similar_miss_counter.values()))
        logging.info('term_similar_count:%s' % sum(self.similar_counter.values()))

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
            self.in_word_category_counter,
            self.minus_suffix_in_word_category_counter,
            self.not_in_vocab_counter,
            self.similar_miss_counter,
            self.similar_counter]):
            _write(records, ('../data/output/cut_result/' + self.model_name + str(i)))

    def register_sug_result(self, word, word_cnt):
        word_cnt[word] += 1

    def load_word_category(self, word_category_path):
        with open(word_category_path, 'r') as f:
            for l in f.readlines():
                yield (l.split('\t')[0].decode('utf8'), l.split('\t')[1].strip().decode('utf8'))

    def sug_by_dict(self, word):
        return self.word_category.get(word, None)

    def sug_by_dict_strip_suffix(self, word):
        return self.word_category.get(re.sub(self.suffix_words, "", word), None)

    def word_in_vocab(self, word):
        return self.model.wv.vocab.has_key(word)

    def sug_by_similar(self, word, topn=10):
        # if word or word-suffix in vocab
        if self.word_in_vocab(word):
            term = word
        elif self.word_in_vocab(re.sub(self.suffix_words, "", word)):
            term = re.sub(self.suffix_words, "", word)
        else:
            return None, None
        # if the word has already addressed
        if self.word_cache.has_key(term):
            return self.word_cache[term][0], self.word_cache[term][1]
        # suggest word by similar word
        similar_words = self.model.most_similar(term, topn=topn)
        category_weights, category_words = defaultdict(int), defaultdict(list)
        # find the most similar category by sum of similar weights
        for i in similar_words:
            if self.word_category.get(i[0], None):
                category_weights[self.word_category.get(i[0])] += i[1]
                category_words[self.word_category.get(i[0])].append(i[0])
        if len(category_weights) > 0:
            c, sim = max(category_weights.iteritems(), key=operator.itemgetter(1))
            info = word + '[' + c + ']' + '#' + '|'.join(category_words[c])
            self.word_cache[term] = (c, info)
            return c, info
        else:
            self.word_cache[term] = (self.similar_miss_info, None)
            return self.similar_miss_info, None

    def sug_flow(self, word, topn=10, mini_sim_ratio=1.0):
        word = word.decode('utf8')
        # Flow1 : suggest by dict
        category = self.sug_by_dict(word)
        if category:
            self.register_sug_result(word, self.in_word_category_counter)
            return category, self.in_word_category_info
        # Flow2 : suggest by dict strip suffix
        category = self.sug_by_dict_strip_suffix(word)
        if category:
            self.register_sug_result(word, self.minus_suffix_in_word_category_counter)
            return category, self.minus_suffix_in_word_category_info
        # Flow3 : suggest by similar word
        category, similar_words = self.sug_by_similar(word, topn)
        if similar_words:
            self.register_sug_result(similar_words, self.similar_counter)
            return category, similar_words.replace(word, '')
        elif category == self.similar_miss_info:
            self.register_sug_result(word, self.similar_miss_counter)
            return None, self.similar_miss_info
        # Flow4 : can not suggest via all the flows above
        self.register_sug_result(word, self.not_in_vocab_counter)
        return None, self.not_in_vocab_info


def sug_by_model(model_dir, model_name):
    word_category_path = '../data/output/domain_dict/snowball_category.csv'
    sug4category = Sug4Category(model_dir + model_name, word_category_path)
    dis_cut_origin_path = '../data/output/cut_result/dis_cut.csv'
    dis_cut_sug_path = '../data/output/cut_result/dis_cut_sug_' + model_name.replace('.model', '') + '.csv'
    all_count, hit_count = 0, 0
    with open(dis_cut_origin_path, 'r') as f_in, open(dis_cut_sug_path, 'w') as f_out:
        for l in f_in.readlines():
            items = l.strip().split(SEG_SPLIT)
            all_count += len(items)
            for i, word in enumerate(items):
                category, reason = sug4category.sug_flow(word)
                if category:
                    hit_count += 1
                    items[i] = ''.join([items[i], '[', category.encode('utf8'), ']'])
                items[i] = ''.join([items[i], '<', reason.encode('utf8'), '>'])
            f_out.write(SEG_SPLIT.join(items) + CLRF)
    logging.info(model_name)
    print('hit / all = %s / %s = %s' % (hit_count, all_count, hit_count * 1.0 / all_count))
    sug4category.display_term_hit_result()
    sug4category.write_term_hitting_record()


class Norm4Category(object):
    def __init__(self, f_path):
        self.category_norm = {k: v for k, v in self.load_category_normalization(f_path)}

    def load_category_normalization(self, f_path):
        with open(f_path, 'r') as f:
            for l in f.readlines():
                items = l.strip().split('\t')
                norm_name, categories = items[0], items[1]
                for category in categories.split(','):
                    yield category.decode('utf8'), norm_name.decode('utf8')

    def norm_term(self, term_category):
        ret = defaultdict(list)
        DUMMY_TERM = 'dummy_term'
        for term, category in term_category.items():
            term = term.decode('utf8')
            if category in set(self.category_norm.keys()):
                ret[self.category_norm[category]].append(term)
            else:
                ret[DUMMY_TERM].append(term)
        return ret


def norm(model_dir, model_name):
    word_category_path = '../data/output/domain_dict/snowball_category.csv'
    sug4category = Sug4Category(model_dir + model_name, word_category_path)
    norm4category = Norm4Category('../data/output/domain_dict/category_normalization.dat')
    # dis_cut_origin_path = '../data/output/cut_result/icd_national_cut.csv'
    dis_cut_origin_path = '../data/output/cut_result/uniq_dis_cut.csv'
    # dis_cut_norm_path = '../data/output/cut_result/icd_national_cut_norm_' + model_name.replace('.model', '') + '.csv'
    dis_cut_norm_path = '../data/output/cut_result/uniq_dis_cut_norm_' + model_name.replace('.model', '') + '.csv'
    with open(dis_cut_origin_path, 'r') as f_in, open(dis_cut_norm_path, 'w') as f_out:
        for l in f_in.readlines():
            items = l.strip().split(SEG_SPLIT)
            term_category = {term: sug4category.sug_flow(term)[0] for term in items}
            norm_result = norm4category.norm_term(term_category)
            if l.strip() != '':
                f_out.write(l.strip() + '\t' + json.dumps(norm_result, ensure_ascii=False).encode('utf8') + CLRF)


if __name__ == '__main__':
    # sug_by_model('../data/input/', 'mc10_s128_w3_cbow.model')
    norm('../data/output/model/', 'mc10_s128_w3_cbow.model')

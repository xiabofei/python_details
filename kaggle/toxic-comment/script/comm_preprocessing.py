import pandas as pd
from data_split import K
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from ipdb import set_trace as st

# Remove all Non-Alpha Numeric and space
special_character_removal = re.compile(r'[^a-z\d ]', re.IGNORECASE)
# Replace all numeric with 'n'
replace_numbers = re.compile(r'\d+', re.IGNORECASE)
# stopwords
stops = set(stopwords.words("english"))

COMMENT_COL = 'comment_text'
ID_COL = 'id'
data_split_dir = '../data/input/data_split/'
data_comm_preprocessed_dir = '../data/input/data_comm_preprocessed/'


# all the words below are included in glove dictionary
# combine these toxic indicators with 'CommProcess.revise_triple_and_more_letters'
toxic_indicator_words = [
    'fuck', 'fucking', 'fucked', 'fuckin', 'fucka', 'fucker', 'fucks', 'fuckers',
    'shit', 'shitty', 'shite',
    'stupid', 'stupids',
    'idiot', 'idiots',
    'suck', 'sucker', 'sucks', 'sucka', 'sucked', 'sucking',
    'ass', 'asshole',
    'gay', 'gays',
]
def _get_toxicIndicator_transformers():
    toxicIndicator_transformers = dict()
    for word in toxic_indicator_words:
        tmp_1 = []
        for c in word:
            if len(tmp_1)>0:
                tmp_2 = []
                for pre in tmp_1:
                    tmp_2.append(pre+c)
                    tmp_2.append(pre+c+c)
                tmp_1 = tmp_2
            else:
                tmp_1.append(c)
                tmp_1.append(c+c)
        toxicIndicator_transformers[word] = tmp_1
    return toxicIndicator_transformers
toxicIndicator_transformers = _get_toxicIndicator_transformers()

# all = 0
# for k,v in toxicIndicator_transformers.items():
#     all += len(v) - 1
# print(all)
# st(context=21)

class CommProcess(object):
    @staticmethod
    def lower(t):
        return t.lower()

    @staticmethod
    def remove_stopwords(t):
        return ' '.join([w for w in t.split() if not w in stops])

    @staticmethod
    def remove_special_chars(t):
        return special_character_removal.sub('', t)

    @staticmethod
    def replace_all_numeric(t):
        return replace_numbers.sub('n', t)

    @staticmethod
    def clean_abbreviation(t):
        t = re.sub(r"\'s", " ", t)
        t = re.sub(r"\'ve", " have ", t)
        t = re.sub(r"can't", "cannot ", t)
        t = re.sub(r"n't", " not ", t)
        t = re.sub(r"i'm", "i am ", t)
        t = re.sub(r"\'re", " are ", t)
        t = re.sub(r"\'d", " would ", t)
        t = re.sub(r"\'ll", " will ", t)
        return t

    @staticmethod
    def revise_triple_and_more_letters(t):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            reg = letter+"{2,}"
            t = re.sub(reg, letter+letter, t)
        return t


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.remove_stopwords,
        CommProcess.clean_abbreviation,
        CommProcess.remove_special_chars,
        CommProcess.replace_all_numeric,
        CommProcess.lower,
        CommProcess.revise_triple_and_more_letters,
    ]
    df[COMMENT_COL].fillna('NA', inplace=True)
    for cp in comm_process_pipeline:
        df[COMMENT_COL] = df[COMMENT_COL].apply(cp)
    return df


if __name__ == '__main__':
    # Process train data in fold
    for k in range(K):
        # raw data
        print('Comm processing in fold {0}'.format(k))
        df_trn = pd.read_csv(data_split_dir + '{0}_train.csv'.format(k))
        df_val = pd.read_csv(data_split_dir + '{0}_valid.csv'.format(k))
        print('  train : {0}'.format(len(df_trn.index)))
        print('  valid : {0}'.format(len(df_val.index)))
        # comm processing
        df_trn = execute_comm_process(df_trn)
        df_val = execute_comm_process(df_val)
        df_trn.to_csv(data_comm_preprocessed_dir + '{0}_train.csv'.format(k), index=False)
        df_val.to_csv(data_comm_preprocessed_dir + '{0}_valid.csv'.format(k), index=False)
    # Process whole train data
    print('Comm processing whole train data')
    df_train = pd.read_csv('../data/input/train.csv')
    df_train = execute_comm_process(df_train)
    df_train.to_csv(data_comm_preprocessed_dir+'train.csv', index=False)
    # Process test data
    print('Comm processing test data')
    df_test = pd.read_csv('../data/input/test.csv')
    df_test = execute_comm_process(df_test)
    df_test.to_csv(data_comm_preprocessed_dir+'test.csv', index=False)

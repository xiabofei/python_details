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

# redundancy words and their right formats
redundancy_rightFormat = {
    'ckckck': 'cock',
    'fuckfuck': 'fuck',
    'lolol': 'lol',
    'lollol': 'lol',
    'pussyfuck': 'fuck',
    'gaygay': 'gay',
    'haha': 'ha',
    'sucksuck': 'suck',
}
redundancy = set(redundancy_rightFormat.keys())

# * mask toxic words // transform to its original format
asterisk_mask = {
    'fuckin': [ 'f**in', 'f***in', 'fuc**n', 'f****n',],
    'fucker': [ 'f***ers', 'f***er', 'f****er',],
    'fucked': [ 'f***ed', 'f**ed', 'f****d',],
    'fuckhead': ['f**khead', '****head', 'headf**k',],
    'fucking': ['fu**ing',],
    'motherfuck': [
        'm*****f*****','mutha******', 'mutha*******','motherf******', 'mother******',
        'mother****er', 'mother****ers', 'motherf***in', 'moderf***n', 'motherf**ker',
        'mother******s', 'mot**rfu*kers', 'motherf**king','motherf**kers',
    ],
    'goddamnit': ['***damnit', ],
    'goddamn': ['g**damn',],
    'damn': ['d**n', ],
    'dumb': ['d**b', ],
    'dicks': ['d***s', 'd**ks',],
    'dick': ['d**k', ],
    'pussy': ['pu***', ],
    'shit': [ 's**t', 'sh**', 's***t',],
    'shithead': ['s**thead'],
    'bullshit': [ 'bull****','bulls***','b***s***', 'bulls**t', ],
    'horseshit': ['horsesh**', ],
    'cunt': ['c**t', ],
    'bitch': ['bi***', 'bit**', 'b**ch', 'b**ch', 'bi*ch', 'b***h', 'b****s', 'b****es', ],
    'asshole': ['as**ole', 'a**h**e', '****holes', '***holes',],
    'jackass': ['jack***', ],
    'sucks': [ 'su**s', 's**ks', 's*cks',],
    'cocksucker': ['co**sucker', ],
    'cocksuckers': ['****suckers', ],
    'bastard': [ 'b**stard', 'b**terd',],
    'niggers': ['ni**ers', 'n***ers', ],
    'niggar': ['nig**', ],
    'hell': ['he**'],
}

mask_origin = dict()
for origin, mask_candidates in asterisk_mask.items():
    for mask in set(mask_candidates):
        mask_origin[mask] = origin
masks = set(mask_origin.keys())
# all the words below are included in glove dictionary
# combine these toxic indicators with 'CommProcess.revise_triple_and_more_letters'
toxic_indicator_words = [
    'fuck', 'fucking', 'fucked', 'fuckin', 'fucka', 'fucker', 'fucks', 'fuckers',
    'fck', 'fcking', 'fcked', 'fckin', 'fcker', 'fcks',
    'fuk', 'fuking', 'fuked', 'fukin', 'fuker', 'fuks', 'fukers',
    'fk', 'fking', 'fked', 'fkin', 'fker', 'fks',
    'shit', 'shitty', 'shite',
    'stupid', 'stupids',
    'idiot', 'idiots',
    'suck', 'sucker', 'sucks', 'sucka', 'sucked', 'sucking',
    'ass', 'asses', 'asshole', 'assholes', 'ashole', 'asholes',
    'gay', 'gays',
    'niga', 'nigga', 'nigar', 'niggar', 'niger', 'nigger',
    'monster', 'monsters',
    'loser', 'losers',
    'nazi', 'nazis',
    'cock', 'cocks', 'cocker', 'cockers',
    'shun',
    'faggot', 'faggy',
    'oh', 'no', 'aw'
]


def _get_toxicIndicator_transformers():
    toxicIndicator_transformers = dict()
    for word in toxic_indicator_words:
        tmp_1 = []
        for c in word:
            if len(tmp_1) > 0:
                tmp_2 = []
                for pre in tmp_1:
                    tmp_2.append(pre + c)
                    tmp_2.append(pre + c + c)
                tmp_1 = tmp_2
            else:
                tmp_1.append(c)
                tmp_1.append(c + c)
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
    def clean_text(t):
        t = re.sub(r"[^A-Za-z0-9,*!?.\/']", " ", t)
        t = replace_numbers.sub(" ", t)
        t = t.lower()
        t = re.sub(r"what's", "what is ", t)
        t = re.sub(r"\'s", " ", t)
        t = re.sub(r"\'ve", " have ", t)
        t = re.sub(r"can't", "cannot ", t)
        t = re.sub(r"n't", " not ", t)
        t = re.sub(r"i'm", "i am ", t)
        t = re.sub(r"\'re", " are ", t)
        t = re.sub(r"\'d", " would ", t)
        t = re.sub(r"\'ll", " will ", t)
        t = re.sub(r",", " ", t)
        t = re.sub(r"\.", " ", t)
        t = re.sub(r"!", " ! ", t)
        t = re.sub(r"\?", " ? ", t)
        t = re.sub(r"\/", " ", t)
        t = re.sub(r"'", " ", t)
        t = re.sub(r" e g ", " eg ", t)
        t = re.sub(r" b g ", " bg ", t)
        t = re.sub(r" u s ", " american ", t)
        return t

    @staticmethod
    def remove_stopwords(t):
        return ' '.join([w for w in t.split() if not w in stops])

    @staticmethod
    def revise_triple_and_more_letters(t):
        for letter in 'abcdefghijklmnopqrstuvwxyz':
            reg = letter + "{2,}"
            t = re.sub(reg, letter + letter, t)
        return t

    @staticmethod
    def revise_redundancy_words(t):
        ret = []
        for word in t.split(' '):
            for redu in redundancy:
                if redu in word:
                    word = redundancy_rightFormat[redu]
                    break
            ret.append(word)
        return ' '.join(ret)

    @staticmethod
    def revise_mask_words(t):
        ret = []
        for word in t.split():
            if word in masks:
                ret.append(mask_origin[word])
            else:
                ret.append(word)
        ret = re.sub(r"\*", " ", ' '.join(ret))
        return ret

    @staticmethod
    def fill_na(t):
        if t.strip() == '':
            return 'NA'
        return t


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.clean_text,
        CommProcess.revise_mask_words,
        CommProcess.remove_stopwords,
        CommProcess.revise_triple_and_more_letters,
        CommProcess.revise_redundancy_words,
        CommProcess.fill_na,
    ]
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
    df_train.to_csv(data_comm_preprocessed_dir + 'train.csv', index=False)
    # Process test data
    print('Comm processing test data')
    df_test = pd.read_csv('../data/input/test.csv')
    df_test = execute_comm_process(df_test)
    df_test.to_csv(data_comm_preprocessed_dir + 'test.csv', index=False)

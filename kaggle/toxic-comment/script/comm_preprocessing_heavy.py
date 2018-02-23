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
data_comm_preprocessed_heavy_dir = '../data/input/data_comm_preprocessed_heavy/'

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
    'fuck': [
        'f**k', 'f***s', 'fu**','fu***', 'f*****k','f**ks', 'f**ck', 'f**kker',
    ],
    'fuckin': [
        'f**kin','f**in', 'f***in', 'fuc**n', 'f****n',
    ],
    'fucker': [
        'f***ers', 'f***er', 'f****er', 'f**ker','f**kers',
    ],
    'fucked': [
        'f***ed', 'f**ed', 'f****d','f**ked',
    ],
    'fuckhead': [
        'f**khead', '****head', 'headf**k',
    ],
    'fucking': [
        'fu**ing', 'f***ing', 'f**cking', 'f**king','f****ng', 'f**ng', 'f*****g','f******g', 'fu*****'
    ],
    'motherfuck': [
        'm*****f*****','mutha******', 'mutha*******','motherf******', 'mother******',
        'mother****er', 'mother****ers', 'motherf***in', 'moderf***n', 'motherf**ker',
        'mother******s', 'mot**rfu*kers', 'mutha***a', 'motherf**king','motherf**kers',
    ],
    'goddamnit': ['***damnit', ],
    'goddamn': ['g**damn',],
    'damn': ['d**n', ],
    'dumb': ['d**b', ],
    'dicks': ['d***s', 'd**ks',],
    'dick': ['d***', 'd**k', ],
    'pussy': ['pu***', ],
    'shit': [
        's**t', 'sh**', 's***t',
    ],
    'shithead': ['s**thead'],
    'bullshit': [
        'bull****','bulls***','b***s***', 'bulls**t',
    ],
    'horseshit': ['horsesh**', ],
    'cunt': ['c**t', ],
    'bitch': [
        'bi***', 'bit**',
        'b**ch', 'b**ch',  'bi*ch', 'b***h', 'b****s',
        'b****es',
    ],
    'asshole': [
        'a**hole', 'a**holes', 'as**ole', 'a**h**e', 'a*****e', '****holes', '***holes',
    ],
    'shithole': ['shit**oles',],
    'ass': ['a**', ],
    'jackass': ['jack***', ],
    'sucks': [
        'su**s', 's**ks', 's*cks',
    ],
    'cocksucker': ['co**sucker', ],
    'cocksuckers': ['****suckers', ],
    'bastard': [
        'b******d', 'b**stard', 'b**terd',
    ],
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
    'shit', 'shitty', 'shite', 'bullshit',
    'stupid', 'stupids',
    'idiot', 'idiots',
    'suck', 'sucker', 'sucks', 'suk', 'suks', 'sucka', 'sucked', 'sucking',
    'ass', 'asses', 'asshole', 'assholes', 'ashole', 'ahole', 'asholes', 'arsehole',
    'gay', 'gays', 'whore', 'hore', 'way',
    'niga', 'nigga', 'niggas', 'nigar', 'niggar', 'niger', 'nigger', 'niggaz',
    'monster', 'monsters', 'burn', 'burned', 'mush',
    'loser', 'losers', 'nazi', 'nazis',
    'cock', 'cocks', 'cocker', 'cockers', 'dick', 'dik',
    'shun', 'faggot', 'fag', 'faggy', 'oh', 'no', 'aw', 'wah', 'ha', 'sh', 'ah', 'uh', 'wo', 'er',
    'mofucka', 'penis', 'penises',
    'vagina', 'cunt', 'cunts',
    'ugly', 'fool', 'pig', 'yay', 'rape', 'raped',
    'moron', 'kill', 'killed', 'garbage', 'bald', 'balding',
    'you', 'erase', 'stop', 'nerd', 'blah', 'prick', 'ball', 'growl', 'growls', 'smelly',
    'vandal', 'so', 'over', 'purger', 'pussy', 'damn',
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

transform_origin = dict()
for origin, toxic_words in toxicIndicator_transformers.items():
    for toxic in toxic_words:
        transform_origin[toxic] = origin
toxic_transforms = set(transform_origin.keys())

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
    def normalize_toxic_word(t):
        ret = []
        for word in t.split():
            if word in toxic_transforms:
                word = transform_origin[word]
            ret.append(word)
        return ' '.join(ret)

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
        if t == '':
            return 'NA'
        return t


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.clean_text,
        CommProcess.remove_stopwords,
        CommProcess.revise_triple_and_more_letters,
        CommProcess.normalize_toxic_word,
        CommProcess.revise_redundancy_words,
        CommProcess.revise_mask_words,
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
        df_trn.to_csv(data_comm_preprocessed_heavy_dir + '{0}_train.csv'.format(k), index=False)
        df_val.to_csv(data_comm_preprocessed_heavy_dir + '{0}_valid.csv'.format(k), index=False)
    # Process whole train data
    print('Comm processing whole train data')
    df_train = pd.read_csv('../data/input/train.csv')
    df_train = execute_comm_process(df_train)
    df_train.to_csv(data_comm_preprocessed_heavy_dir + 'train.csv', index=False)
    # Process test data
    print('Comm processing test data')
    df_test = pd.read_csv('../data/input/test.csv')
    df_test = execute_comm_process(df_test)
    df_test.to_csv(data_comm_preprocessed_heavy_dir + 'test.csv', index=False)

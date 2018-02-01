import pandas as pd
from data_split import K
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

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


def execute_comm_process(df):
    comm_process_pipeline = [
        CommProcess.remove_stopwords,
        CommProcess.remove_special_chars,
        CommProcess.replace_all_numeric,
        CommProcess.lower,
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

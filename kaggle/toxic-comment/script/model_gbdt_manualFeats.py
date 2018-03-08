# encoding=utf8
import argparse

import pandas as pd
import numpy as np

import lightgbm as lgbm

from data_split import label_candidates
from data_split import K
from comm_preprocessing import COMMENT_COL, ID_COL

data_split_dir = '../data/input/data_split/'

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix

from lgbm_params import params_groups
from scipy.sparse import hstack

import re
import string

from ipdb import set_trace as st
import gc

n_classes = len(label_candidates)

NAN_BLANK = 'nanblank'
NUM_OF_LABEL = 6

# lgbm params
num_boost_round=1000
early_stopping_rounds=50
verbose_eval=20

# Contraction replacement patterns
cont_patterns = [
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(regex), repl) for (regex, repl) in cont_patterns]

def prepare_for_char_n_gram(text):
    clean = bytes(text.lower(), encoding="utf-8")
    clean = clean.replace(b"\n", b" ")
    clean = clean.replace(b"\t", b" ")
    clean = clean.replace(b"\b", b" ")
    clean = clean.replace(b"\r", b" ")
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])
    clean = re.sub(b"\d+", b" ", clean)
    clean = re.sub(b'\s+', b' ', clean)
    clean = re.sub(b'\s+$', b'', clean)
    clean = re.sub(b" ", b"# #", clean)
    clean = b"#" + clean + b"#"
    return str(clean, 'utf-8')

def count_regexp_occ(regexp="", text=None):
    return len(re.findall(regexp, text))

def get_indicators_and_clean_comments(df):
    df["ant_slash_n"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\n", x))
    df["raw_word_len"] = df["comment_text"].apply(lambda x: len(x.split()))
    df["raw_char_len"] = df["comment_text"].apply(lambda x: len(x))
    df["nb_upper"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
    df["nb_fk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
    df["nb_sk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
    df["nb_dk"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
    df["nb_you"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
    df["nb_mother"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
    df["nb_ng"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\Wnigger\W", x))
    df["start_with_columns"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"^\:+", x))
    df["has_timestamp"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
    df["has_date_long"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
    df["has_date_short"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
    df["has_http"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
    df["has_mail"] = df["comment_text"].apply(
        lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
    )
    df["has_emphasize_equal"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
    df["has_emphasize_quotes"] = df["comment_text"].apply(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))
    df["clean_comment"] = df["comment_text"].apply(lambda x: prepare_for_char_n_gram(x))
    df["clean_word_len"] = df["clean_comment"].apply(lambda x: len(x.split()))
    df["clean_char_len"] = df["clean_comment"].apply(lambda x: len(x))
    df["clean_chars"] = df["clean_comment"].apply(lambda x: len(set(x)))
    df["clean_chars_ratio"] = df["clean_comment"].apply(lambda x: len(set(x))) / df["clean_comment"].apply(
        lambda x: 1 + min(99, len(x)))

def char_analyzer(text):
    tokens = text.split()
    return [token[i: i + 3] for token in tokens for i in range(len(token) - 2)]


def read_test_data():
    df_test = pd.read_csv('../data/input/test.csv')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    df_train = pd.read_csv('../data/input/train.csv')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


def get_gbdt_params(label_name):
    assert label_name in label_candidates, 'wrong label name'
    return params_groups[label_name]

def get_idx_trn_val(ids):
    # get each fold trn/val data ids [(trn, val),...]
    ids_trn_val = []
    for k in range(K):
        df_trn = pd.read_csv(data_split_dir + '{0}_train.csv'.format(k))
        df_val = pd.read_csv(data_split_dir + '{0}_valid.csv'.format(k))
        ids_trn = df_trn[ID_COL].values.tolist()
        ids_val = df_val[ID_COL].values.tolist()
        ids_trn_val.append((ids_trn, ids_val))

    # get each fold trn/val data index of array
    idx_trn_val = []
    for ids_trn, ids_val in ids_trn_val:
        ids_trn = set(ids_trn)
        ids_val = set(ids_val)
        idx_trn = [idx for idx, id in enumerate(ids) if id in ids_trn]
        idx_val = [idx for idx, id in enumerate(ids) if id in ids_val]
        idx_trn_val.append((idx_trn, idx_val))

    return idx_trn_val

def run_cv():
    # read train/test data
    df_train = read_train_data()
    id_train = df_train[ID_COL].values.tolist()
    y_true = df_train[label_candidates].values

    df_test = read_test_data()
    id_test = df_test[ID_COL].values.tolist()

    get_indicators_and_clean_comments(df=df_train)
    get_indicators_and_clean_comments(df=df_test)

    num_features = [f_ for f_ in df_train.columns
                    if f_ not in ["comment_text", "clean_comment", "id", "remaining_chars",
                                  'has_ip_address'] + label_candidates]
    skl = MinMaxScaler()
    train_num_features = csr_matrix(skl.fit_transform(df_train[num_features]))
    test_num_features = csr_matrix(skl.fit_transform(df_test[num_features]))

    # Get TF-IDF features
    train_text = df_train['clean_comment']
    test_text = df_test['clean_comment']
    all_text = pd.concat([train_text, test_text])

    # First on real words
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 2),
        max_features=20000)
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(train_text)
    test_word_features = word_vectorizer.transform(test_text)
    del word_vectorizer
    gc.collect()

    # Now use the char_analyzer to get another TFIDF
    char_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        tokenizer=char_analyzer,
        analyzer='word',
        ngram_range=(1, 1),
        max_features=50000)
    char_vectorizer.fit(all_text)
    train_char_features = char_vectorizer.transform(train_text)
    test_char_features = char_vectorizer.transform(test_text)

    del char_vectorizer
    gc.collect()

    X_train_word_char = hstack([train_char_features, train_word_features, train_num_features]).tocsr()
    X_test_word_char = hstack([test_char_features, test_word_features, test_num_features]).tocsr()

    # get idx of trn/val for each fold
    print('Getting array index of train/valid for each fold')
    idx_trn_val = get_idx_trn_val(id_train)

    # preds on test/valid
    preds_test = np.zeros((X_test_word_char.shape[0], n_classes))
    preds_valid = np.zeros((X_train_word_char.shape[0], n_classes))

    # cv and train/predict
    for label_col, label_name in enumerate(label_candidates):
        print('\nlabel column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        # cv best boost rounds
        train_set = lgbm.Dataset(data=X_train_word_char, label=y_true[:, label_col])
        params = get_gbdt_params(label_name)
        print('lgbm params : {0}'.format(params))
        hist = lgbm.cv(
            params=params,
            train_set=train_set,
            folds=idx_trn_val,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            metrics=['auc'],
            verbose_eval=verbose_eval,
        )
        bst_boost_rounds = np.argmax(hist['auc-mean']) + 1
        bst_acc_score = np.max(hist['auc-mean'])
        print('label column : {0}'.format(label_col))
        print('label name : {0}'.format(label_name))
        print('best boost rounds {0}, best auc score {1}'.format(bst_boost_rounds, bst_acc_score))

        # oof train and predict
        for fold, (idx_trn, idx_val) in enumerate(idx_trn_val):
            print('fold {0}'.format(fold))
            print('    train')
            dat_trn = lgbm.Dataset(data=X_train_word_char[idx_trn, :], label=y_true[idx_trn, label_col])
            model = lgbm.train(
                params=params,
                train_set=dat_trn,
                num_boost_round=bst_boost_rounds,
                verbose_eval=verbose_eval,
            )
            print('    predict')
            preds_valid[idx_val, label_col] = model.predict(
                data=X_train_word_char[idx_val,:], num_iteration=bst_boost_rounds)
            preds_test[:,label_col] = model.predict(data=X_test_word_char, num_iteration=bst_boost_rounds) / K
            del model

    # ensemble cv score
    score = roc_auc_score(y_true=y_true, y_score=preds_valid, average='macro')
    print('\ncv score : {0}'.format(score))
    # divide data for ensemble
    for fold, (_, idx_val) in enumerate(idx_trn_val):
        preds_valid_fold = preds_valid[idx_val,:].T
        df_preds_val = pd.DataFrame()
        idx_val_set = set(idx_val)
        df_preds_val[ID_COL] = [ id for idx, id in enumerate(id_train) if idx in idx_val_set ]
        for idx, label in enumerate(label_candidates):
            df_preds_val[label] = preds_valid_fold[idx]
        df_preds_val.to_csv('../data/output/preds/gbdt/{0}fold_valid.csv'.format(fold), index=False)

    # record ensemble result
    preds_test = preds_test.T
    df_preds_test = pd.DataFrame()
    df_preds_test[ID_COL] = id_test
    for idx, label in enumerate(label_candidates):
        df_preds_test[label] = preds_test[idx]
    df_preds_test.to_csv('../data/output/preds/gbdt/avg_submit.csv', index=False)

if __name__ == '__main__':
    run_cv()

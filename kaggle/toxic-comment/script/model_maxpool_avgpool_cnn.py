'''
1) use glove 100d for word embedding layer
2) use gru as rnn layer
'''
import argparse

import pandas as pd
import numpy as np
from numpy import asarray

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import SpatialDropout1D
from keras.layers import concatenate

from keras.models import Model
from keras.optimizers import Nadam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_split import label_candidates
from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL
from comm_preprocessing import toxicIndicator_transformers

from roc_auc_metric import RocAucMetricCallback
from roc_auc_metric import VAL_AUC

MAX_NUM_WORDS = 283000  # keras Tokenizer keep MAX_NUM_WORDS-1 words and left index 0 for null word
MAX_SEQUENCE_LENGTH = 200
RUNS_IN_FOLD = 5
NUM_OF_LABEL = 6

EPOCHS = 30
BATCH_SIZE = 64

from ipdb import set_trace as st
import gc

# mode = 'try'
mode = 'other'


def read_data_in_fold(k):
    df_trn = pd.read_csv(data_comm_preprocessed_dir + '{0}_train.csv'.format(k))
    df_val = pd.read_csv(data_comm_preprocessed_dir + '{0}_valid.csv'.format(k))
    print('train data in fold {0} : {1}'.format(k, len(df_trn.index)))
    print('valid data in fold {0} : {1}'.format(k, len(df_val.index)))
    return df_trn, df_val


def read_test_data():
    df_test = pd.read_csv(data_comm_preprocessed_dir + 'test.csv')
    df_test[COMMENT_COL] = df_test[COMMENT_COL].astype('str')
    print('test data {0}'.format(len(df_test.index)))
    return df_test


def read_train_data():
    df_train = pd.read_csv(data_comm_preprocessed_dir + 'train.csv')
    df_train[COMMENT_COL] = df_train[COMMENT_COL].astype('str')
    print('train data {0}'.format(len(df_train.index)))
    return df_train


def get_fitted_tokenizer(df_train, df_test):
    comments_train = df_train[COMMENT_COL].values.tolist()
    comments_test = df_test[COMMENT_COL].values.tolist()
    # remain '!' and '?'
    tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(comments_train + comments_test)
    return tokenizer


def get_padded_sequence(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequence


def get_embedding_lookup_table(word_index, embedding_path, embedding_dim):
    def _get_embedding_index(path):
        ret = dict()
        for l in open(path):
            values = l.split(' ')
            word = values[0]
            vector = asarray(values[1:], dtype='float32')
            ret[word] = vector
        print('total {0} word vectors'.format(len(ret)))
        # add toxic transformer vector
        for toxic, transformers in toxicIndicator_transformers.items():
            if toxic in ret.keys():
                for transformer in transformers:
                    ret[transformer] = ret[toxic]
        print('total {0} word vectors after add toxic indicator transformers'.format(len(ret)))
        return ret

    nb_words = min(MAX_NUM_WORDS, len(word_index))
    if mode == 'try':
        return np.zeros((nb_words, embedding_dim))

    print('! index : {0}'.format(word_index['!']))
    print('? index : {0}'.format(word_index['?']))

    # get glove word vector
    embedding_index = _get_embedding_index(embedding_path)
    embedding_lookup_table = np.zeros((nb_words, embedding_dim))
    for word, index in word_index.items():
        if index >= MAX_NUM_WORDS:
            continue
        vector = embedding_index.get(word)
        if vector is not None:
            embedding_lookup_table[index] = vector
    print('null word embeddings : {0}'.format(
        np.sum(np.sum(embedding_lookup_table, axis=1) == 0)))
    del embedding_index
    gc.collect()

    return embedding_lookup_table


def get_model(embedding_lookup_table, dropout, spatialDropout):
    print('dropout : {0}'.format(dropout))
    print('spatial dropout : {0}'.format(spatialDropout))
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(
        input_dim=embedding_lookup_table.shape[0],
        output_dim=embedding_lookup_table.shape[1],
        weights=[embedding_lookup_table],
        trainable=False
    )(input_layer)
    glove_embedding_layer = SpatialDropout1D(spatialDropout)(embedding_layer)
    kernel_sizes = [2, 3, 4]
    num_filters = 256
    multi_filters = []
    for kernel_size in kernel_sizes:
        conv1d = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(glove_embedding_layer)
        bn = BatchNormalization()(conv1d)
        max_pool = GlobalMaxPooling1D()(bn)
        # avg_pool = GlobalAveragePooling1D()(bn)
        multi_filters.append(max_pool)
        # multi_filters.append(avg_pool)

    layer = concatenate(multi_filters)
    layer = Dropout(dropout)(layer)
    output_layer = Dense(6, activation='sigmoid')(layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(), metrics=['acc'])
    return model


def run_one_fold(fold):
    # read whole train / test data for tokenizer
    df_train = read_train_data()
    df_test = read_test_data()

    # fit tokenizer
    tokenizer = get_fitted_tokenizer(df_train, df_test)
    word_index = tokenizer.word_index
    transformers_count = 0
    all_words = set(word_index.keys())
    for toxic, transformers in toxicIndicator_transformers.items():
        for transformer in transformers:
            if transformer == toxic:
                continue
            if transformer in all_words:
                transformers_count += tokenizer.word_counts[transformer]
    print('toxic transformer count : {0}'.format(transformers_count))
    print('unique token : {0}'.format(len(word_index)))

    # get embedding lookup table
    embedding_dim = 300
    embedding_path = '../data/input/glove_dir/glove.840B.300d.txt'
    # embedding_path = '../data/input/fasttext_dir/fasttext.300d.txt'
    embedding_lookup_table = get_embedding_lookup_table(word_index, embedding_path, embedding_dim)

    # read in fold data
    df_trn, df_val = read_data_in_fold(fold)

    # prepare data
    X_test = get_padded_sequence(tokenizer, df_test[COMMENT_COL].astype('str').values.tolist())
    id_test = df_test[ID_COL].values.tolist()
    print('Test data shape {0}'.format(X_test.shape))

    X_trn = get_padded_sequence(tokenizer, df_trn[COMMENT_COL].astype('str').values.tolist())
    y_trn = df_trn[label_candidates].values
    print('Fold {0} train data shape {1} '.format(fold, X_trn.shape))

    X_val = get_padded_sequence(tokenizer, df_val[COMMENT_COL].astype('str').values.tolist())
    y_val = df_val[label_candidates].values
    id_val = df_val[ID_COL].values.tolist()
    print('Fold {0} valid data shape {1} '.format(fold, X_val.shape))

    # preds result array
    preds_test = np.zeros((X_test.shape[0], NUM_OF_LABEL))
    preds_valid = np.zeros((X_val.shape[0], NUM_OF_LABEL))

    # train model
    for run in range(RUNS_IN_FOLD):
        print('\nFold {0} run {1} begin'.format(fold, run))

        # model
        model = get_model(embedding_lookup_table, float(FLAGS.dp), float(FLAGS.sdp))
        # print(model.summary())

        if mode == 'try':
            st(context=3)

        # callbacks
        val_auc = RocAucMetricCallback()
        es = EarlyStopping(monitor=VAL_AUC, mode='max', patience=5)
        bst_model_path = \
            '../data/output/model/{0}fold_{1}run_{2}dp_{3}sdp_maxpool_avgpool_cnn.h5'.format(
                fold, run, FLAGS.dp, FLAGS.sdp)
        mc = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        rp = ReduceLROnPlateau(
            monitor=VAL_AUC, mode='max',
            patience=3,
            factor=np.sqrt(0.1),
            verbose=1
        )
        # train
        hist = model.fit(
            x=X_trn, y=y_trn,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            callbacks=[val_auc, es, mc, rp]
        )
        model.load_weights(bst_model_path)
        bst_val_score = max(hist.history[VAL_AUC])
        print('\nFold {0} run {1} best val score : {2}'.format(fold, run, bst_val_score))

        # predict
        preds_test += model.predict(X_test, batch_size=512, verbose=1) / RUNS_IN_FOLD
        preds_valid += model.predict(X_val, batch_size=512, verbose=1) / RUNS_IN_FOLD
        print('\nFold {0} run {1} done'.format(fold, run))

        del model
        gc.collect()

    # record preds result
    preds_test = preds_test.T
    df_preds_test = pd.DataFrame()
    df_preds_test[ID_COL] = id_test
    for idx, label in enumerate(label_candidates):
        df_preds_test[label] = preds_test[idx]
    df_preds_test.to_csv(
        '../data/output/preds/maxpool_avgpool_cnn/{0}/{1}/{2}fold_test.csv'.format(
            FLAGS.dp, FLAGS.sdp, fold), index=False)

    preds_valid = preds_valid.T
    df_preds_val = pd.DataFrame()
    df_preds_val[ID_COL] = id_val
    for idx, label in enumerate(label_candidates):
        df_preds_val[label] = preds_valid[idx]
    df_preds_val.to_csv(
        '../data/output/preds/maxpool_avgpool_cnn/{0}/{1}/{2}fold_valid.csv'.format(
            FLAGS.dp, FLAGS.sdp, fold), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='train on which fold')
    parser.add_argument('--dp', type=str, default='0.55', help='dropout')
    parser.add_argument('--sdp', type=str, default='0.4', help='spatial dropout')
    FLAGS, _ = parser.parse_known_args()
    run_one_fold(FLAGS.fold)

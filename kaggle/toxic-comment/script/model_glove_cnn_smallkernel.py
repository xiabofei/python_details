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
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Flatten
from keras.layers import Merge, Concatenate
from keras.layers import Activation
from keras.regularizers import l2
from keras.layers import CuDNNGRU
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Bidirectional
from keras.layers import BatchNormalization

from keras.models import Model
from keras.optimizers import RMSprop, Nadam, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_split import label_candidates
from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL
from comm_preprocessing import toxicIndicator_transformers

from kmax_pooling import KMaxPooling

import pickle

MAX_NUM_WORDS = 380000  # keras Tokenizer keep MAX_NUM_WORDS-1 words and left index 0 for null word
MAX_SEQUENCE_LENGTH = 100
RUNS_IN_FOLD = 5
NUM_OF_LABEL = 6

EPOCHS = 30
BATCH_SIZE = 128

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
    tokenizer = Tokenizer()
    # tokenizer.num_words = MAX_NUM_WORDS
    tokenizer.fit_on_texts(comments_train + comments_test)
    return tokenizer


def get_padded_sequence(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequence


def get_embedding_lookup_table(word_index, glove_path, fasttext_path, embedding_dim):
    def _get_glove_embedding_index(path):
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

    def _get_fasttext_embedding_index(path):
        ret = dict()
        for l in open(path):
            values = l.strip().split(' ')
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
    fasttext_embedding_lookup_table = np.zeros((nb_words, embedding_dim))
    '''
    # get fasttext word vector
    fasttext_embedding_index = _get_fasttext_embedding_index(fasttext_path)
    for word, index in word_index.items():
        if index >= MAX_NUM_WORDS:
            continue
        vector = fasttext_embedding_index.get(word)
        if vector is not None:
            fasttext_embedding_lookup_table[index] = vector
    print('fasttext null word embeddings : {0}'.format(
        np.sum(np.sum(fasttext_embedding_lookup_table, axis=1) == 0)))
    del fasttext_embedding_index
    gc.collect()
    '''

    # get glove word vector
    glove_embedding_index = _get_glove_embedding_index(glove_path)
    glove_embedding_lookup_table = np.zeros((nb_words, embedding_dim))
    for word, index in word_index.items():
        if index >= MAX_NUM_WORDS:
            continue
        vector = glove_embedding_index.get(word)
        if vector is not None:
            glove_embedding_lookup_table[index] = vector
    print('glove null word embeddings : {0}'.format(
        np.sum(np.sum(glove_embedding_lookup_table, axis=1) == 0)))
    del glove_embedding_index
    gc.collect()
    return glove_embedding_lookup_table, fasttext_embedding_lookup_table


def get_model(glove_embedding_lookup_table, fasttext_embedding_lookup_table):
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    ## Chanel 1 : glove embedding
    glove_embedding_layer = Embedding(
        input_dim=glove_embedding_lookup_table.shape[0],
        output_dim=glove_embedding_lookup_table.shape[1],
        weights=[glove_embedding_lookup_table],
        trainable=False
    )(input_layer)
    glove_layer = glove_embedding_layer
    kernel_sizes = [2,3,4]
    num_filters = 256
    glove_conv_maxpools = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(glove_layer)
        conv = BatchNormalization()(conv)
        maxpool = KMaxPooling(5)(conv)
        # maxpool = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH-kernel_size+1)(conv)
        glove_conv_maxpools.append(maxpool)

    ## Chanel 2 : fasttext embedding
    '''
    fasttext_embedding_layer = Embedding(
        input_dim=fasttext_embedding_lookup_table.shape[0],
        output_dim=fasttext_embedding_lookup_table.shape[1],
        weights=[fasttext_embedding_lookup_table],
        trainable=False
    )(input_layer)
    fasttext_layer = fasttext_embedding_layer
    kernel_sizes = [5,6,7]
    num_filters = 256
    fasttext_conv_maxpools = []
    for kernel_size in kernel_sizes:
        conv = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu')(fasttext_layer)
        conv = BatchNormalization()(conv)
        maxpool = MaxPooling1D(pool_size=MAX_SEQUENCE_LENGTH-kernel_size+1)(conv)
        fasttext_conv_maxpools.append(maxpool)

    '''
    # merge glove convs and fasttext convs
    layer = Concatenate(axis=1)(glove_conv_maxpools)
    # fasttext_concatenated_tensor = Concatenate(axis=1)(fasttext_conv_maxpools)
    # merge = Merge(mode='max')([glove_concatenated_tensor, fasttext_concatenated_tensor])
    # merge = Concatenate(axis=1)(glove_conv_maxpools + fasttext_conv_maxpools)
    layer = Flatten()(layer)
    layer = Dropout(0.5)(layer)
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
            if transformer==toxic:
                continue
            if transformer in all_words:
                transformers_count += tokenizer.word_counts[transformer]
                # print(transformer)
    print('toxic transformer count : {0}'.format(transformers_count))
    print('unique token : {0}'.format(len(word_index)))

    # get embedding lookup table
    embedding_dim = 300
    glove_path = '../data/input/glove_dir/glove.840B.300d.txt'
    fasttext_path = '../data/input/fasttext_dir/fasttext.300d.txt'
    glove_embedding_lookup_table, fasttext_embedding_lookup_table = \
        get_embedding_lookup_table(word_index, glove_path, fasttext_path, embedding_dim)

    # read in fold data
    df_trn, df_val = read_data_in_fold(fold)

    # prepare data
    X_test = get_padded_sequence(tokenizer, df_test[COMMENT_COL].values.tolist())
    id_test = df_test[ID_COL].values.tolist()
    print('Test data shape {0}'.format(X_test.shape))

    X_trn = get_padded_sequence(tokenizer, df_trn[COMMENT_COL].values.tolist())
    y_trn = df_trn[label_candidates].values
    print('Fold {0} train data shape {1} '.format(fold, X_trn.shape))

    X_val = get_padded_sequence(tokenizer, df_val[COMMENT_COL].values.tolist())
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
        model = get_model(glove_embedding_lookup_table, fasttext_embedding_lookup_table)
        print(model.summary())

        if mode == 'try':
            st(context=3)

        # callbacks
        es = EarlyStopping(monitor='val_acc', mode='max', patience=8)
        bst_model_path = '../data/output/model/{0}fold_{1}run_glove_cnn.h5'.format(fold, run)
        mc = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        rp = ReduceLROnPlateau(
            monitor='val_acc', mode='max',
            patience=2,
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
            callbacks=[es, mc, rp]
        )
        model.load_weights(bst_model_path)
        bst_val_score = max(hist.history['val_acc'])
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
    df_preds_test.to_csv('../data/output/preds/glove_cnn/{0}fold_test.csv'.format(fold), index=False)

    preds_valid = preds_valid.T
    df_preds_val = pd.DataFrame()
    df_preds_val[ID_COL] = id_val
    for idx, label in enumerate(label_candidates):
        df_preds_val[label] = preds_valid[idx]
    df_preds_val.to_csv('../data/output/preds/glove_cnn/{0}fold_valid.csv'.format(fold), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='train on which fold')
    FLAGS, _ = parser.parse_known_args()
    run_one_fold(FLAGS.fold)

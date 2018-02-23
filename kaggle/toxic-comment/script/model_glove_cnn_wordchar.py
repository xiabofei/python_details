'''
1) use glove 100d for word embedding layer
2) use gru as rnn layer
'''
import argparse

import pandas as pd
import numpy as np
from numpy import asarray

from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPool2D, Flatten
from keras.layers import Concatenate, Add
from keras.layers import Activation
from keras.regularizers import l2
from keras.layers import CuDNNGRU
from keras.layers import Input
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Bidirectional
from keras.layers import BatchNormalization
from keras.layers import Maximum

from keras.models import Model
from keras.optimizers import RMSprop, Nadam, Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_split import label_candidates
from comm_preprocessing import data_comm_preprocessed_dir
from comm_preprocessing import COMMENT_COL, ID_COL
from comm_preprocessing import toxicIndicator_transformers

from attention_layer import Attention

MAX_NUM_WORDS = 284540  # keras Tokenizer keep MAX_NUM_WORDS-1 words and left index 0 for null word
MAX_SEQUENCE_LENGTH = 200
MAX_SEQUENCE_LENGTH_CHAR = 1000
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
    tokenizer = Tokenizer(filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(comments_train + comments_test)
    return tokenizer

def get_fitted_tokenizer_charLevel(df_train, df_test):
    comments_train = df_train[COMMENT_COL].values.tolist()
    comments_test = df_test[COMMENT_COL].values.tolist()
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(comments_train + comments_test)
    return tokenizer

def get_padded_sequence(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequence

def get_padded_sequence_charLevel(tokenizer, texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH_CHAR)
    return padded_sequence


def get_embedding_lookup_table(word_index, word_index_char, glove_path, embedding_dim):
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
    # get glove embedding index
    glove_embedding_index = _get_glove_embedding_index(glove_path)

    # word level
    nb_words = min(MAX_NUM_WORDS, len(word_index))
    print('! index : {0}'.format(word_index['!']))
    print('? index : {0}'.format(word_index['?']))
    glove_embedding_lookup_table = np.zeros((nb_words, embedding_dim))
    for word, index in word_index.items():
        if index >= MAX_NUM_WORDS:
            continue
        vector = glove_embedding_index.get(word)
        if vector is not None:
            glove_embedding_lookup_table[index] = vector
    print('glove null word embeddings : {0}'.format(np.sum(np.sum(glove_embedding_lookup_table, axis=1) == 0)))
    gc.collect()

    # select char level vector using glove
    nb_chars = len(word_index_char) + 1
    print('! index : {0}'.format(word_index_char['!']))
    print('? index : {0}'.format(word_index_char['?']))
    glove_embedding_lookup_table_char = np.zeros((nb_chars, embedding_dim))
    for char, index in word_index_char.items():
        vector = glove_embedding_index.get(char)
        if vector is not None:
            glove_embedding_lookup_table_char[index] = vector
    print('glove null char embeddings : {0}'.format(np.sum(np.sum(glove_embedding_lookup_table_char, axis=1) == 0)))
    return glove_embedding_lookup_table, glove_embedding_lookup_table_char


def get_model(glove_embedding_lookup_table, glove_embedding_lookup_table_char, dropout):
    print('dropout : {0}'.format(dropout))
    ## word level
    glove_embedding_layer = Embedding(
        input_dim=glove_embedding_lookup_table.shape[0],
        output_dim=glove_embedding_lookup_table.shape[1],
        weights=[glove_embedding_lookup_table],
        trainable=False
    )
    embedding_dim_word = glove_embedding_lookup_table.shape[1]

    input_layer_word = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    word_embedding_layer = glove_embedding_layer(input_layer_word)
    filter_sizes = [3,4,5]
    num_filters = 128
    reshape = Reshape((MAX_SEQUENCE_LENGTH, embedding_dim_word, 1))(word_embedding_layer)
    wordLevel_multi_filters = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim_word), padding='valid',
                    kernel_initializer='normal', activation='relu')(reshape)
        bn = BatchNormalization()(conv)
        maxpool = MaxPool2D(
            pool_size=(MAX_SEQUENCE_LENGTH - filter_size + 1, 1), strides=(1, 1), padding='valid')(bn)
        wordLevel_multi_filters.append(maxpool)

    ## char level
    glove_embedding_layer_char = Embedding(
        input_dim=glove_embedding_lookup_table_char.shape[0],
        output_dim=glove_embedding_lookup_table_char.shape[1],
        weights=[glove_embedding_lookup_table_char],
        trainable=True
    )
    embedding_dim_char = glove_embedding_lookup_table_char.shape[1]
    input_layer_char = Input(shape=(MAX_SEQUENCE_LENGTH_CHAR,), dtype='int32')
    char_embedding_layer = glove_embedding_layer_char(input_layer_char)
    filter_sizes = [2,3,4]
    num_filters = 128
    reshape = Reshape((MAX_SEQUENCE_LENGTH_CHAR, embedding_dim_char, 1))(char_embedding_layer)
    charLevel_multi_filters = []
    for filter_size in filter_sizes:
        conv = Conv2D(num_filters, kernel_size=(filter_size, embedding_dim_char), padding='valid',
                      kernel_initializer='normal', activation='tanh')(reshape)
        maxpool = MaxPool2D(
            pool_size=(MAX_SEQUENCE_LENGTH_CHAR - filter_size + 1, 1), strides=(1, 1), padding='valid')(conv)
        charLevel_multi_filters.append(maxpool)

    # merge 'word Level cnn' and 'char Level cnn' together
    layer = Concatenate(axis=1)(wordLevel_multi_filters + charLevel_multi_filters)
    layer = BatchNormalization()(layer)
    layer = Flatten()(layer)
    layer = Dropout(dropout)(layer)
    output_layer = Dense(6, activation='sigmoid')(layer)
    model = Model(inputs=[input_layer_word, input_layer_char], outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer=Nadam(), metrics=['acc'])
    return model


def run_one_fold(fold):
    # read whole train / test data for tokenizer
    df_train = read_train_data()
    df_test = read_test_data()

    # fit tokenizer : word level
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

    # fit tokenizer : char level
    tokenizer_char = get_fitted_tokenizer_charLevel(df_train, df_test)
    word_index_char = tokenizer_char.word_index
    print('unique token char : {0}'.format(len(word_index_char)))

    # get embedding lookup table word level / char level
    embedding_dim = 300
    glove_path = '../data/input/glove_dir/glove.840B.300d.txt'
    glove_embedding_lookup_table, glove_embedding_lookup_table_char = \
        get_embedding_lookup_table(word_index, word_index_char, glove_path, embedding_dim)

    # read in fold data
    df_trn, df_val = read_data_in_fold(fold)

    # prepare data : word level
    X_test_word = get_padded_sequence(tokenizer, df_test[COMMENT_COL].astype('str').values.tolist())
    id_test = df_test[ID_COL].values.tolist()
    print('Test data shape {0}'.format(X_test_word.shape))
    X_trn_word = get_padded_sequence(tokenizer, df_trn[COMMENT_COL].astype('str').values.tolist())
    y_trn = df_trn[label_candidates].values
    print('Fold {0} train data shape {1} '.format(fold, X_trn_word.shape))
    X_val_word = get_padded_sequence(tokenizer, df_val[COMMENT_COL].astype('str').values.tolist())
    y_val = df_val[label_candidates].values
    id_val = df_val[ID_COL].values.tolist()
    print('Fold {0} valid data shape {1} '.format(fold, X_val_word.shape))

    # prepare data : char level
    X_test_char = get_padded_sequence_charLevel(tokenizer_char, df_test[COMMENT_COL].astype('str').values.tolist())
    id_test = df_test[ID_COL].values.tolist()
    print('Test data shape {0}'.format(X_test_char.shape))
    X_trn_char = get_padded_sequence_charLevel(tokenizer_char, df_trn[COMMENT_COL].astype('str').values.tolist())
    y_trn = df_trn[label_candidates].values
    print('Fold {0} train data shape {1} '.format(fold, X_trn_char.shape))
    X_val_char = get_padded_sequence_charLevel(tokenizer_char, df_val[COMMENT_COL].astype('str').values.tolist())
    y_val = df_val[label_candidates].values
    id_val = df_val[ID_COL].values.tolist()
    print('Fold {0} valid data shape {1} '.format(fold, X_val_char.shape))

    # prepare word / char level data
    X_test = [X_test_word, X_test_char]
    X_trn = [X_trn_word, X_trn_char]
    X_val = [X_val_word, X_val_char]


    # preds result array
    preds_test = np.zeros((X_test_word.shape[0], NUM_OF_LABEL))
    preds_valid = np.zeros((X_val_word.shape[0], NUM_OF_LABEL))

    # train model
    for run in range(RUNS_IN_FOLD):
        print('\nFold {0} run {1} begin'.format(fold, run))

        # model
        model = get_model(glove_embedding_lookup_table, glove_embedding_lookup_table_char, float(FLAGS.dp))
        # print(model.summary())

        if mode == 'try':
            st(context=3)

        # callbacks
        es = EarlyStopping(monitor='val_acc', mode='max', patience=5)
        bst_model_path = '../data/output/model/{0}fold_{1}run_glove_cnn.h5'.format(fold, run)
        mc = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        rp = ReduceLROnPlateau(
            monitor='val_acc', mode='max',
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
    df_preds_test.to_csv('../data/output/preds/glove_cnn/{0}/{1}fold_test.csv'.format(FLAGS.dp, fold), index=False)

    preds_valid = preds_valid.T
    df_preds_val = pd.DataFrame()
    df_preds_val[ID_COL] = id_val
    for idx, label in enumerate(label_candidates):
        df_preds_val[label] = preds_valid[idx]
    df_preds_val.to_csv('../data/output/preds/glove_cnn/{0}/{1}fold_valid.csv'.format(FLAGS.dp, fold), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=str, default='0', help='train on which fold')
    parser.add_argument('--dp', type=str, default='0.5', help='dropout')
    FLAGS, _ = parser.parse_known_args()
    run_one_fold(FLAGS.fold)

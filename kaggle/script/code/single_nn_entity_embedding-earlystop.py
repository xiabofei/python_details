'''
    This script provides code for training a neural network with entity embeddings
    of the 'cat' variables. For more details on entity embedding, see:
    https://github.com/entron/entity-embedding-rossmann

    8-Fold training with 3 averaged runs per fold. Results may improve with more folds & runs.
'''

import numpy as np
import pandas as pd
# random seeds for stochastic parts of neural network
np.random.seed(2017)
from tensorflow import set_random_seed
set_random_seed(2017)

from keras.models import Sequential
from keras.layers import Dense, Activation, Merge, Reshape, Dropout
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import os

from sklearn.model_selection import StratifiedKFold
from evaluation import gini_score_keras
from io_utils import comm_skf, Number_of_folds
from fe import Processer

from logging_manage import initialize_logger
import logging
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')
# Data loading & preprocessing
df_train = pd.read_csv('../../data/input/train.csv')
df_train.drop([149161], axis=0, inplace=True)
tmp = pd.DataFrame(data=df_train.values, columns=df_train.columns, index=None)
df_train = tmp
df_test = pd.read_csv('../../data/input/test.csv')

X_train, y_train = df_train.iloc[:, 2:], df_train.target
X_test = df_test.iloc[:, 1:]

X_train = Processer.negative_one_vals(X_train)
X_test = Processer.negative_one_vals(X_test)


cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))]

X_train = X_train[cols_use]
X_test = X_test[cols_use]


VALUE_FEA_NUM = 25

col_vals_dict = {c: list(X_train[c].unique()) for c in X_train.columns if c.endswith('_cat')}

embed_cols = []
for c in col_vals_dict:
    if len(col_vals_dict[c]) > 2:
        embed_cols.append(c)
        print(c + ': %d values' % len(col_vals_dict[c]))  # look at value counts to know the embedding dimensions

print('\n')


def build_embedding_network(
        dropout_one=0.35,
        dropout_two=0.15,
        dropout_three=0.15,
        lr=0.001):
    models = []

    model_ps_ind_02_cat = Sequential()
    model_ps_ind_02_cat.add(Embedding(5, 3, input_length=1))
    model_ps_ind_02_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_ind_02_cat)

    model_ps_ind_04_cat = Sequential()
    model_ps_ind_04_cat.add(Embedding(3, 2, input_length=1))
    model_ps_ind_04_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_ind_04_cat)

    model_ps_ind_05_cat = Sequential()
    model_ps_ind_05_cat.add(Embedding(8, 5, input_length=1))
    model_ps_ind_05_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_ind_05_cat)

    model_ps_car_01_cat = Sequential()
    model_ps_car_01_cat.add(Embedding(13, 7, input_length=1))
    model_ps_car_01_cat.add(Reshape(target_shape=(7,)))
    models.append(model_ps_car_01_cat)

    model_ps_car_02_cat = Sequential()
    model_ps_car_02_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_02_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_02_cat)

    model_ps_car_03_cat = Sequential()
    model_ps_car_03_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_03_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_03_cat)

    model_ps_car_04_cat = Sequential()
    model_ps_car_04_cat.add(Embedding(10, 5, input_length=1))
    model_ps_car_04_cat.add(Reshape(target_shape=(5,)))
    models.append(model_ps_car_04_cat)

    model_ps_car_05_cat = Sequential()
    model_ps_car_05_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_05_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_05_cat)

    model_ps_car_06_cat = Sequential()
    model_ps_car_06_cat.add(Embedding(18, 8, input_length=1))
    model_ps_car_06_cat.add(Reshape(target_shape=(8,)))
    models.append(model_ps_car_06_cat)

    model_ps_car_07_cat = Sequential()
    model_ps_car_07_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_07_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_07_cat)

    model_ps_car_09_cat = Sequential()
    model_ps_car_09_cat.add(Embedding(6, 3, input_length=1))
    model_ps_car_09_cat.add(Reshape(target_shape=(3,)))
    models.append(model_ps_car_09_cat)

    model_ps_car_10_cat = Sequential()
    model_ps_car_10_cat.add(Embedding(3, 2, input_length=1))
    model_ps_car_10_cat.add(Reshape(target_shape=(2,)))
    models.append(model_ps_car_10_cat)

    model_ps_car_11_cat = Sequential()
    model_ps_car_11_cat.add(Embedding(104, 10, input_length=1))
    model_ps_car_11_cat.add(Reshape(target_shape=(10,)))
    models.append(model_ps_car_11_cat)

    model_rest = Sequential()
    model_rest.add(Dense(16, input_dim=VALUE_FEA_NUM))
    models.append(model_rest)

    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(80))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_one))
    model.add(Dense(20))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_two))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_three))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # model.compile(loss='binary_crossentropy', optimizer='adam')
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=['auc'])

    return model


# converting data to list format to match the network structure
def preproc(X_train, X_val, X_test):
    input_list_train = []
    input_list_val = []
    input_list_test = []

    # the cols to be embedded: rescaling to range [0, # values)
    for c in embed_cols:
        raw_vals = np.unique(X_train[c])
        val_map = {}
        for i in range(len(raw_vals)):
            val_map[raw_vals[i]] = i
        input_list_train.append(X_train[c].map(val_map).values)
        input_list_val.append(X_val[c].map(val_map).fillna(0).values)
        input_list_test.append(X_test[c].map(val_map).fillna(0).values)

    # the rest of the columns
    other_cols = [c for c in X_train.columns if (not c in embed_cols)]
    input_list_train.append(X_train[other_cols].values)
    input_list_val.append(X_val[other_cols].values)
    input_list_test.append(X_test[other_cols].values)

    return input_list_train, input_list_val, input_list_test


# gini scoring function from kernel at:
# https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n


def gini_normalizedc(a, p):
    # return ginic(a, p) / ginic(a, a)
    return gini_score_keras(a, p)


# network training
K = 10
runs_per_fold = 5
n_epochs = 30
BATCH_SIZE = 8192
skf = StratifiedKFold(n_splits=K, random_state=2017, shuffle=True)

# params setting
# param_distribution = dict(
#     dropout_one=[0.302499],
#     dropout_two=[0.09532],
#     dropout_three=[0.151212],
#     activation = ['relu']
# )
param_distribution = dict(
    dropout_one=0.303928,
    dropout_two=0.184395,
    dropout_three=0.2433226,
)
# execute random setting
# param in this iteration
dropout_one = param_distribution['dropout_one']
dropout_two = param_distribution['dropout_two']
dropout_three = param_distribution['dropout_three']
logging.info('dropout_one : {0}'.format(dropout_one))
logging.info('dropout_two : {0}'.format(dropout_two))
logging.info('dropout_three : {0}'.format(dropout_three))
# cv folds
cv_ginis = []
full_val_preds = np.zeros(np.shape(X_train)[0])
y_preds = np.zeros((np.shape(X_test)[0], K))
for i, (f_ind, outf_ind) in enumerate(skf.split(X_train, y_train)):
    # fold i
    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    X_test_f = X_test.copy()
    pos = (pd.Series(y_train_f == 1))
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]
    proc_X_train_f, proc_X_val_f, proc_X_test_f = preproc(X_train_f, X_val_f, X_test_f)
    val_preds = 0
    for j in range(runs_per_fold):
        # fold i run j
        # logging.info('Fold {0} Run {1} begin'.format(i, j))
        early_stopping =EarlyStopping(monitor='val_loss', patience=10)
        bst_model_path = './tmp/nn.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
        NN = build_embedding_network(
            dropout_one=dropout_one,
            dropout_two=dropout_two,
            dropout_three=dropout_three,
        )
        hist = NN.fit(
            x=proc_X_train_f,
            y=y_train_f.values,
            validation_data=(proc_X_val_f, y_val_f.values),
            epochs=n_epochs,
            batch_size=BATCH_SIZE,
            callbacks=[early_stopping, model_checkpoint],
            verbose=0,
        )
        NN.load_weights(bst_model_path)
        os.remove(bst_model_path)
        val_preds += NN.predict(proc_X_val_f, batch_size=65536)[:, 0] / runs_per_fold
        y_preds[:, i] += NN.predict(proc_X_test_f, batch_size=65536)[:, 0] / runs_per_fold
        del NN
        gc.collect()
    full_val_preds[outf_ind] += val_preds
    cv_gini = gini_normalizedc(y_val_f.values, val_preds)
    cv_ginis.append(cv_gini)
    logging.info('[{0}] Fold {1} prediction cv gini: {2}'.format(BATCH_SIZE, i, cv_gini))
    gc.collect()

mean_oof_gini = np.mean(cv_ginis)
full_validation_gini = gini_normalizedc(y_train.values, full_val_preds)

logging.info('[{0}] Mean out of fold gini: {1}'.format(BATCH_SIZE, mean_oof_gini))
logging.info('[{0}] Full validation gini: {1}'.format(BATCH_SIZE, full_validation_gini))
y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id': df_test.id,
                       'target': y_pred_final},
                      columns=['id', 'target'])

df_sub.to_csv('../../data/nn/nn-sub-earlystop.csv', index=False)

pd.DataFrame(full_val_preds).to_csv('../../data/nn/nn-val-earlystop.csv', index=False)


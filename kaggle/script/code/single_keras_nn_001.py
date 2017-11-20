# encoding=utf8

import numpy as np
np.random.seed(20)
import pandas as pd

from evaluation import gini_score_keras

from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from io_utils import Number_of_folds, comm_skf
from logging_manage import initialize_logger
import logging
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')
from ipdb import set_trace as st

'''Data loading & preprocessing
'''

X_train = pd.read_csv('../../data/input/train.csv')
# have to do this, otherwise mess with keras gpu process
X_train.drop([149161], axis=0, inplace=True)
tmp = pd.DataFrame(data=X_train.values, columns=X_train.columns, index=None)
X_train = tmp
X_test = pd.read_csv('../../data/input/test.csv')

X_train, y_train = X_train.iloc[:, 2:], X_train.target
X_test, test_id = X_test.iloc[:, 1:], X_test.id

# OHE / some feature engineering adapted from the1owl kernel at:
# https://www.kaggle.com/the1owl/forza-baseline/code

# excluded columns based on snowdog's old school nn kernel at:
# https://www.kaggle.com/snowdog/old-school-nnet

X_train['negative_one_vals'] = np.sum((X_train == -1).values, axis=1)
X_test['negative_one_vals'] = np.sum((X_test == -1).values, axis=1)

to_drop = ['ps_car_11_cat', 'ps_ind_14', 'ps_car_11', 'ps_car_14', 'ps_ind_06_bin',
           'ps_ind_09_bin', 'ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin',
           'ps_ind_13_bin']

cols_use = [c for c in X_train.columns if (not c.startswith('ps_calc_'))
            & (not c in to_drop)]

X_train = X_train[cols_use]
X_test = X_test[cols_use]

one_hot = {c: list(X_train[c].unique()) for c in X_train.columns}

# note that this encodes the negative_one_vals column as well
for c in one_hot:
    if len(one_hot[c]) > 2 and len(one_hot[c]) < 105:
        for val in one_hot[c]:
            newcol = c + '_oh_' + str(val)
            X_train[newcol] = (X_train[c].values == val).astype(np.int)
            X_test[newcol] = (X_test[c].values == val).astype(np.int)
        # X_train.drop(labels=[c], axis=1, inplace=True)
        # X_test.drop(labels=[c], axis=1, inplace=True)

X_train = X_train.replace(-1, np.NaN)  # Get rid of -1 while computing interaction col
X_test = X_test.replace(-1, np.NaN)

X_train['ps_car_13_x_ps_reg_03'] = X_train['ps_car_13'] * X_train['ps_reg_03']
X_test['ps_car_13_x_ps_reg_03'] = X_test['ps_car_13'] * X_test['ps_reg_03']

X_train = X_train.fillna(-1)
X_test = X_test.fillna(-1)
logging.info('X_train shape : {0}'.format(X_train.shape))
logging.info('X_test shape : {0}'.format(X_test.shape))

'''Gini scoring function
'''


# gini scoring function from kernel at:
# https://www.kaggle.com/tezdhar/faster-gini-calculation
def ginic(actual, pred):
    n = len(actual)
    a_s = actual[np.argsort(pred)]
    a_c = a_s.cumsum()
    giniSum = a_c.sum() / a_c[-1] - (n + 1) / 2.0
    return giniSum / n


def gini_normalizedc(a, p):
    return ginic(a, p) / ginic(a, a)


'''5-fold neural network training
'''

BAGGING_PER_FOLD = 3  # bagging on each fold
EPOCHS = 15
BATCH_SIZE = 4096
BATCH_SIZE_FEED = 65536
VERBOSE = 1
skf = comm_skf
early_stopping =EarlyStopping(monitor='loss', patience=10)

cv_ginis = []
y_preds = np.zeros((np.shape(X_test)[0], Number_of_folds))

def get_nn_model():
    NN = Sequential()
    NN.add(Dense(35, activation='relu', input_dim=np.shape(X_train_f)[1]))
    NN.add(Dropout(0.3))
    NN.add(Dense(1, activation='sigmoid'))
    NN.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return NN

for i, (f_ind, outf_ind) in enumerate(skf.split(X_train, y_train)):

    logging.info('Fold {0} begin'.format(i))

    X_train_f, X_val_f = X_train.loc[f_ind].copy(), X_train.loc[outf_ind].copy()
    y_train_f, y_val_f = y_train[f_ind], y_train[outf_ind]
    logging.info('before upsampling train shape {0}'.format(X_train_f.shape))

    # upsampling adapted from kernel:
    # https://www.kaggle.com/ogrellier/xgb-classifier-upsampling-lb-0-283
    pos = (pd.Series(y_train_f == 1))
    #
    # Add positive examples
    X_train_f = pd.concat([X_train_f, X_train_f.loc[pos]], axis=0)
    y_train_f = pd.concat([y_train_f, y_train_f.loc[pos]], axis=0)
    logging.info('after upsampling train shape {0}'.format(X_train_f.shape))

    # Shuffle data
    idx = np.arange(len(X_train_f))
    np.random.shuffle(idx)
    X_train_f = X_train_f.iloc[idx]
    y_train_f = y_train_f.iloc[idx]

    # track oof bagged prediction for cv scores
    val_preds = 0

    for j in range(BAGGING_PER_FOLD):
        set_random_seed(1000 * i + j)
        NN = get_nn_model()
        NN.fit(
            x=X_train_f.values,
            y=y_train_f.values,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            shuffle=True,
            callbacks=[early_stopping]
        )
        logging.info('Fold {0} Run {1} Results'.format(i, j))
        val_pred = NN.predict(X_val_f.values, batch_size=BATCH_SIZE_FEED)[:, 0]
        y_preds[:, i] += NN.predict(X_test.values, batch_size=BATCH_SIZE_FEED)[:, 0] / BAGGING_PER_FOLD
        val_gini = gini_score_keras(y_val_f.values, val_pred)
        logging.info('Validation gini: {0}'.format(val_gini))
        val_preds += val_pred / BAGGING_PER_FOLD

    cv_ginis.append(val_gini)
    logging.info('Fold {0} prediction cv gini: {1}'.format(i, val_gini))

logging.info('Mean oof gini: {0}'.format(np.mean(cv_ginis)))
y_pred_final = np.mean(y_preds, axis=1)

df_sub = pd.DataFrame({'id': test_id,
                       'target': y_pred_final},
                      columns=['id', 'target'])
df_sub.to_csv('NNShallow_5fold_3runs_sub.csv', index=False)
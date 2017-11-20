# encoding=utf8
import numpy as np
np.random.seed(2017)
from tensorflow import set_random_seed
set_random_seed(2017)

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import os

from fe import Processer, Compose

from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from evaluation import GiniEvaluation, gini_score, gini_score_keras
from io_utils import read_data_nn, comm_skf, write_data, Number_of_folds, check_nan
from sklearn.preprocessing import Imputer
from fe import FeatureImportance
from scipy.stats import randint, uniform
from logging_manage import initialize_logger
import logging
import gc

from ipdb import set_trace as st

## logging setting
initialize_logger(output_dir='../../data/log/')
## Data Loading
df_train, df_y, df_test, df_sub, train_id = read_data_nn()

'''
# Data transform
transformer = [
    (Processer.drop_columns, dict(col_names=df_train.columns[df_train.columns.str.startswith('ps_calc_')])),
    (Processer.drop_columns, dict(col_names=['ps_ind_10_bin', 'ps_ind_11_bin', 'ps_ind_12_bin', 'ps_ind_13_bin'])),
    (Processer.negative_one_vals, dict()),
    (Processer.median_mean_range, dict(opt_median=True, opt_mean=False)),
    (Processer.convert_reg_03, dict()),
    (Processer.dtype_transform, dict()),
]

# Execute transforms pipeline
logging.info('Transform train data')
df_train = Compose(transformer)(df_train)
logging.info('Transform test data')
df_test = Compose(transformer)(df_test)
# Execute ohe
df_train, df_test = Processer.ohe(df_train, df_test, [a for a in df_train.columns if a.endswith('cat')])


## Handle missing value
# select value and category columns
value_cols = [c for c in df_train.columns if '_cat' not in c and '_bin' not in c ]
cat_cols = [c for c in df_train.columns if c not in value_cols ]
# handle category columns
logging.info('Handle category feature columns missing values')
imputer_for_cat = Imputer(strategy='most_frequent', axis=0)
df_train[cat_cols] = imputer_for_cat.fit_transform(X=df_train[cat_cols].values)
df_test[cat_cols] = imputer_for_cat.fit_transform(X=df_test[cat_cols].values)
# handle value columns
logging.info('Handle value feature columns missing values')
imputer_for_value = Imputer(strategy='median', axis=0)
df_train[value_cols] = imputer_for_value.fit_transform(X=df_train[value_cols].values)
df_test[value_cols] = imputer_for_value.fit_transform(X=df_test[value_cols].values)
# normalize value columns for nn input
df_train = Processer.normalization(df_train, df_train.columns)
df_test = Processer.normalization(df_test, df_train.columns)
# save memory
df_train = Processer.dtype_transform(df_train)
df_test = Processer.dtype_transform(df_test)
gc.collect()
# write file to local disk
df_train.to_csv('../../data/input/df_train_4_nn.csv', index=False)
df_test.to_csv('../../data/input/df_test_4_nn.csv', index=False)
'''


# Read data from local disk file
logging.info('loading already processed train from disk file')
df_train = pd.read_csv('../../data/input/df_train_4_nn.csv', index_col=None)
logging.info('loading already processed test from disk file')
df_test = pd.read_csv('../../data/input/df_test_4_nn.csv', index_col=None)
# Make sure no NaN or Inf in train or test data
assert check_nan(df_train) == 0, 'df_train contain nan value'
assert check_nan(df_test) == 0, 'df_test contain nan value'
# save memory
df_train = Processer.dtype_transform(df_train)
df_test = Processer.dtype_transform(df_test)

## Common skf
skf = comm_skf

# feature and label for train
X = df_train.values
y = df_y.values

gc.collect()


def build_model(
        hidden_sizes_one=32,
        dropout_one=0.5,
        activation_one='relu',
        hidden_sizes_two=16,
        dropout_two=0.5,
        activation_two='relu',
        optimizer='adam'):
    NN = Sequential()
    NN.add(Dense(hidden_sizes_one, activation=activation_one, input_dim=X.shape[1]))
    NN.add(Dropout(dropout_one))
    NN.add(Dense(hidden_sizes_two, activation=activation_two))
    NN.add(Dropout(dropout_two))
    NN.add(Dense(1, activation='sigmoid'))
    NN.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return NN


# nn.summary()

nn_params = dict(
    dropout_one=0.3,
    dropout_two=0.3,
    verbose=1,
    epochs=10,
)
fc_classifier = KerasClassifier(build_model, **nn_params)

SEARCH = 1


def choose_param_distribution(search):
    nn_param_distribution = dict()
    if search == 1:
        nn_param_distribution = dict(
            activation_one=['sigmoid'],
            activation_two=['sigmoid'],
            batch_size=[128, 256, 512],
            hidden_sizes_one=[128, 256],
            hidden_sizes_two=[128, 256],
        )
    if search == 2:
        nn_param_distribution = dict(
            activation_one=['relu'],
            activation_two=['tanh'],
            batch_size=[128, 512, 1024],
            hidden_sizes_one=[128, 256],
            hidden_sizes_two=[128, 256],
        )
    if search == 3:
        nn_param_distribution = dict(
            activation_one=['tanh'],
            activation_two=['tanh'],
            batch_size=[128, 512, 1024],
            hidden_sizes_one=[128, 256],
            hidden_sizes_two=[128, 256],
        )
    if search == 4:
        nn_param_distribution = dict(
            activation_one=['tanh'],
            activation_two=['sigmoid'],
            batch_size=[128, 512, 1024],
            hidden_sizes_one=[128, 256],
            hidden_sizes_two=[128, 256],
        )
    return nn_param_distribution


def param_search(nn_param_distribution, if_random=True):
    logging.info('SEARCH space'.format(nn_param_distribution))
    logging.info('random search begin')
    if if_random:
        gs = RandomizedSearchCV(
            estimator=fc_classifier,
            param_distributions=nn_param_distribution,
            n_iter=20,
            cv=comm_skf,
            scoring=make_scorer(gini_score, greater_is_better=True, needs_proba=True),
            verbose=2,
        )
    else:
        gs = GridSearchCV(
            estimator=fc_classifier,
            param_grid=nn_param_distribution,
            cv=comm_skf,
            scoring=make_scorer(gini_score, greater_is_better=True, needs_proba=True),
            verbose=2,
        )
    return gs

'''
nn_gs = param_search(choose_param_distribution(SEARCH), if_random=False)
nn_gs.fit(X=X, y=y)
logging.info('best score {0}, best params {1}'.format(nn_gs.best_score_, nn_gs.best_params_))
for score in nn_gs.grid_scores_:
    logging.info(score)
'''
#################################################################################
#################################################################################


#################################################################################
PREDICT_BATCH_SIZE = 65536

stacker_train = np.zeros((X.shape[0], 1))

model_param_oof = dict(
    hidden_sizes_one=256,
    dropout_one=0.3,
    activation_one='tanh',
    hidden_sizes_two=128,
    activation_two='tanh',
    dropout_two=0.3,
    # optimizer=SGD(lr=0.01, momentum=0.9),
    # optimizer=Adam(),
    optimizer='adam',
)
fit_param_oof = dict(
    epochs = 100,
    batch_size = 2048
)

for index, (trn_idx, val_idx) in enumerate(comm_skf.split(X, y)):

    trn_x, val_x = X[trn_idx], X[val_idx]
    trn_y, val_y = y[trn_idx], y[val_idx]

    logging.info('Train model in fold {0}'.format(index))
    bst_model_path = './tmp/single_nn_001.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    early_stopping =EarlyStopping(monitor='val_loss', patience=10)
    model = build_model(**model_param_oof)
    history = model.fit(
        x=trn_x,
        y=trn_y,
        batch_size=fit_param_oof['batch_size'],
        epochs=fit_param_oof['epochs'],
        validation_data=(val_x, val_y),
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint],
        verbose=1,
    )
    logging.info('model train history : {0}'.format(history.history))

    model.load_weights(bst_model_path)
    os.remove(bst_model_path)

    bst_score_pret = min(history.history['val_loss'])
    logging.info('Fold {0} finished, best val-loss: {1}'.format(index, bst_score_pret))

    stacker_train[val_idx] = model.predict(val_x, batch_size=PREDICT_BATCH_SIZE)
    df_sub['target'] += model.predict(df_test.values, batch_size=PREDICT_BATCH_SIZE)[:,0] / Number_of_folds

    logging.info('gini on valid fold : {0}'.format(gini_score_keras(val_y, stacker_train[val_idx,0])))

logging.info('{0} of folds'.format(Number_of_folds))
logging.info('Oof by nn model Done')
logging.info('gini score on local cv : {0}'.format(gini_score_keras(y, stacker_train)))

write_data(
    df_sub=df_sub,
    stacker_train=stacker_train,
    train_id=train_id,
    sub_filename='sub_single_nn_001_test.csv',
    train_filename='single_nn_001_train.csv'
)
logging.info('keras nn done')
#################################################################################

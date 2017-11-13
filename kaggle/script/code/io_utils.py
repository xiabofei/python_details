# encoding=utf8
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import os

def read_data():
    root_dir = '../../data/input/'
    df_train = pd.read_csv(os.path.join(root_dir, 'train.csv'), na_values=-1)
    df_train.drop([149161], axis=0, inplace=True)
    df_test = pd.read_csv(os.path.join(root_dir, 'test.csv'), na_values=-1)
    df_y = df_train['target']
    train_id = df_train['id']
    df_train.drop(['id', 'target'], axis=1, inplace=True)
    df_sub = df_test['id'].to_frame()
    df_sub['target'] = 0.0
    df_test.drop(['id'], axis=1, inplace=True)
    return df_train, df_y, df_test, df_sub, train_id

Number_of_folds = 5
comm_skf = StratifiedKFold(n_splits=Number_of_folds, shuffle=True, random_state=2017)


def write_data(df_sub, train_id, stacker_train, sub_filename, train_filename):
    # create submit file
    df_sub.to_csv('../../data/for_stacker/' + sub_filename, index=False)
    s_train = pd.DataFrame()
    s_train['id'] = train_id
    s_train['prob'] = stacker_train
    s_train.to_csv('../../data/for_stacker/' + train_filename, index=False)

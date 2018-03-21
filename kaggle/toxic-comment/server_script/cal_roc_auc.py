# encoding=utf8
import pandas as pd
import numpy as np
from data_split import K
from data_split import label_candidates
from comm_preprocessing import ID_COL
from sklearn.metrics import roc_auc_score

from ipdb import set_trace as st

n_classes = len(label_candidates)

root_dir = '../data/output/preds/deep_gru/0.375/0.4/'
# root_dir = '../data/output/preds/pool_cnn/0.35/0.4/'
# root_dir = '../data/output/preds/maxpool_deep_cnn/0.45/0.4/'
#root_dir = '../data/output/preds/maxpool_gru/0.3/0.35/'

# get id and true label
df_train = pd.read_csv('../data/input/train.csv')
ids = df_train[ID_COL].values.tolist()
y_true = df_train[label_candidates].values

# get oof valid preds
id_preds = {}
for k in range(K):
    print('fold {0}'.format(k))
    df = pd.read_csv(root_dir+'{0}fold_valid.csv'.format(k))
    id_list = df[ID_COL].values.tolist()
    preds_list = df[label_candidates].values.tolist()
    for id, preds in zip(id_list, preds_list):
        id_preds[id] = preds

# match valid preds and origin label by id
y_score = []
for id in ids:
    y_score.append(id_preds[id])
y_score = np.array(y_score)

# mean column-wise roc auc
score = roc_auc_score(y_true=y_true, y_score=y_score, average='macro')
print('roc auc score : {0}'.format(score))

# check average='macro'
score = 0.0
for i, label in enumerate(label_candidates):
    auc = roc_auc_score(y_true=y_true[:,i], y_score=y_score[:,i])
    print('label {0} : auc {1}'.format(label, auc))
    score += auc / n_classes
print('average 6 roc auc score : {0}'.format(score))

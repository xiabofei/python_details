#######################################
## load data
#######################################

import cPickle
import numpy as np

def load_from_pkl(file_path):
    x_bow = []
    y_label = []
    raw_data = cPickle.load(open(file_path,'r'))
    for i,data in enumerate(raw_data):
        x_bow.append(data['data'])
        y_label.append(data['label'])
    return np.array(x_bow), np.array(y_label)  

x, y = load_from_pkl('./4bow/bow.pkl')

#######################################
## xgboost
#######################################

import xgboost as xgb
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

N = 20
skf = StratifiedKFold(n_splits=N, shuffle=True)

params = {
    # logistic regression for binary classification, output probability
    'objective':'binary:logistic',
    # evaluation metrics for validation data
    'eval_metric':'logloss',
    # step size shrinkage used in update to prevents overfitting
    'eta':0.01,
    # XGBoost randomly collected how much data instances to grow trees to prevents overfitting
    'subsample':0.8,
    # subsample ratio of columns when constructing each tree
    'colsample_bytree':0.3,
    #  maximum depth of a tree
    'max_depth':6,
    # the minimum weight/counts of instances in a leaf node, larger prevents overfitting
    'min_child_weight':8,
    # number of parallel threads used to run xgboost
    'nthread':8,
    # if printing running messages
    'silent':1,
    # L1 regularization term on weights
    'alpha':0.001,
    # minimum loss reduction required to make a further partition on a leaf node of the tree larger prevent overfitting
    # 'gamma':0.01,
    # random number seed
    'seed':np.random.randint(0,99999)
}

## cross validation
folds = skf.split(x, y)
dtrain = xgb.DMatrix(data = x, label = y)
num_boost_round = 10000
bst = xgb.cv(
    params,
    dtrain,
    num_boost_round,
    N,
    folds=folds,
    early_stopping_rounds = 100,
    verbose_eval=100
)

best_rounds = np.argmin(bst['test-logloss-mean'])
train_score = bst['train-logloss-mean'][best_rounds]
best_val_score = bst['test-logloss-mean'][best_rounds]

print('Best train_loss: %.5f, val_loss: %.5f at round %d.'% \
        (train_score, best_val_score, best_rounds))

## out-of-fold prediction
oof_preds = np.zeros(y.shape)
for trn_idx, val_idx in skf.split(x, y):
    trn_x, val_x = x[trn_idx], x[val_idx]
    trn_y = y[trn_idx]
        
    dtrn = xgb.DMatrix(data=trn_x, label=trn_y)
    dval = xgb.DMatrix(data=val_x)
        
    cv_model = xgb.train(params, dtrn, best_rounds)
    oof_preds[val_idx] = cv_model.predict(dval)
    
## search for threshold
best_threshold = 0
best_score = 0
for i in range(100):
    score = f1_score(y, oof_preds > (i/100.0))
    if score > best_score:
        best_threshold = i/100.0
        best_score = score

print('Best threshold: %.2f'%(best_threshold))
print('Best f1 score: %.5f'%best_score)
print('XGBoost done')

#######################################
## svm
#######################################

"""
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict

svc_model = SVC(C=4.0, probability=True)
oofs = cross_val_predict(svc_model, x, y, cv=20, n_jobs=8, method='predict_proba')
oof_preds = oofs[:, 1].ravel()

best_threshold = 0
best_score = 0
for i in range(100):
    score = f1_score(y, oof_preds > (i/100.0))
    if score > best_score:
        best_threshold = i/100.0
        best_score = score

print('Best threshold: %.2f'%(best_threshold))
print('Best f1 score: %.5f'%best_score)
print('SVM done')
"""


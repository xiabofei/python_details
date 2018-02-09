###################################################
## import packages
###################################################

import sys
import pandas as pd
import numpy as np

from scipy.sparse import hstack

from sklearn import model_selection
from sklearn import linear_model
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import CountVectorizer

###################################################
## load train data
###################################################

trainDF = pd.read_csv('./train.csv')
trainDF.ix[trainDF['question1'].isnull(),['question1']] = 'random empty question'
trainDF.ix[trainDF['question2'].isnull(),['question2']] = 'random empty question'

testDF = pd.read_csv('./test.csv')
testDF.ix[testDF['question1'].isnull(),['question1']] = 'random empty question'
testDF.ix[testDF['question2'].isnull(),['question2']] = 'random empty question'

###################################################
## set parameters
###################################################

max_features = np.random.randint(275000, 325000)
ngram_range = (1, np.random.randint(8, 13))
C = 0.05 + 0.1 * np.random.random()

letter_sequences = (sys.argv[1] == 'True')
feature_type_1 = (sys.argv[2] == 'True')

if letter_sequences:
    min_df = np.random.randint(7, 14)
    ngram_range = (1, ngram_range[1] - 4)

STAMP = '%d_%d_%d_%.2f_%s_%s'%(max_features, min_df, ngram_range[1], C, letter_sequences, feature_type_1)
print(STAMP)

###################################################
## extract features
###################################################

print('Start feature extraction')

if letter_sequences:
    extractor = CountVectorizer(max_df=0.999, min_df=min_df, max_features=max_features, \
            analyzer='char', ngram_range=ngram_range, \
            binary=True, lowercase=True)
else:
    extractor = CountVectorizer(max_df=0.999, min_df=min_df, max_features=max_features, \
            analyzer='word', ngram_range=ngram_range, stop_words='english', \
            binary=True, lowercase=True)

extractor.fit(pd.concat((trainDF.ix[:,'question1'],trainDF.ix[:,'question2'])).unique())

trainQuestion1_BOW_rep = extractor.transform(trainDF.ix[:,'question1'])
trainQuestion2_BOW_rep = extractor.transform(trainDF.ix[:,'question2'])
lables = np.array(trainDF.ix[:,'is_duplicate'])

if feature_type_1:
    X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int)
else:
    X = -(trainQuestion1_BOW_rep != trainQuestion2_BOW_rep).astype(int) + \
        trainQuestion1_BOW_rep.multiply(trainQuestion2_BOW_rep)

idx_stg1 = np.load('stg1_idx.npy')
idx_stg2 = np.load('stg2_idx.npy')

X_stg1 = X[idx_stg1]
y_stg1 = lables[idx_stg1]
X_stg2 = X[idx_stg2]
y_stg2 = lables[idx_stg2]

testQuestion1_BOW_rep = extractor.transform(testDF.ix[:,'question1'])
testQuestion2_BOW_rep = extractor.transform(testDF.ix[:,'question2'])

if feature_type_1:
    X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int)
else:
    X_test = -(testQuestion1_BOW_rep != testQuestion2_BOW_rep).astype(int) + \
        testQuestion1_BOW_rep.multiply(testQuestion2_BOW_rep)

###################################################
## load side data
###################################################

stg1_side = np.load('stg1_side.npy')
stg2_side = np.load('stg2_side.npy')
test_side = np.load('test_side.npy')

###################################################
## train the model
###################################################

print('Start model training')
logisticRegressor = linear_model.LogisticRegression(C=C, solver='sag', n_jobs=8)
logisticRegressor.fit(hstack((X_stg1, stg1_side)), y_stg1)

###################################################
## make submission
###################################################

print('Start making the submission for stage 2')
preds_stg2 = logisticRegressor.predict_proba(hstack((X_stg2, stg2_side)))[:,1]
val_loss = log_loss(y_stg2, preds_stg2)
print('Validation log loss: %.5f'%val_loss)
out_df = pd.DataFrame({'is_duplicate': preds_stg2.ravel()})
out_df.to_csv('./stg2/%s_%s/%.5f_'%(letter_sequences, feature_type_1, val_loss)+STAMP+'.csv', index=False)

print('Start making the submission for test set')
seperators= [750000,1500000]
preds_test1 = logisticRegressor.predict_proba(hstack((X_test[:seperators[0],:], test_side[:seperators[0],:])))[:,1]
preds_test2 = logisticRegressor.predict_proba(hstack((X_test[seperators[0]:seperators[1],:], test_side[seperators[0]:seperators[1],:])))[:,1]
preds_test3 = logisticRegressor.predict_proba(hstack((X_test[seperators[1]:,:], test_side[seperators[1]:,:])))[:,1]
preds_test = np.hstack((preds_test1, preds_test2, preds_test3))
out_df = pd.DataFrame({'is_duplicate': preds_test.ravel()})
out_df.to_csv('./test/%s_%s/%.5f_'%(letter_sequences, feature_type_1, val_loss)+STAMP+'.csv', index=False)

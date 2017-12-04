# encoding=utf8

#################################################################################
import math
import numpy as np
import subprocess
import pandas as pd
from sklearn.model_selection import KFold
from evaluation import gini_score_libffm
from io_utils import comm_skf, Number_of_folds
from logging_manage import initialize_logger
import logging
import os
import gc

from ipdb import set_trace as st

initialize_logger(output_dir='../../data/log/')
#################################################################################

#################################################################################
train = pd.read_csv('../../data/input/train.csv')
test = pd.read_csv('../../data/input/test.csv')
test.insert(1,'target',0)
logging.info('train shape {0}'.format(train.shape))
logging.info('test shape {0}'.format(test.shape))
#################################################################################

#################################################################################
x = pd.concat([train,test])
x = x.reset_index(drop=True)
unwanted = x.columns[x.columns.str.startswith('ps_calc_')]
x.drop(unwanted,inplace=True,axis=1)
#################################################################################

#################################################################################
features = x.columns[2:]
categories = []
for c in features:
    trainno = len(x.loc[:train.shape[0],c].unique())
    testno = len(x.loc[train.shape[0]:,c].unique())
    logging.info('column {0}, train no {1}, test no {2}'.format(c,trainno,testno))
#################################################################################

#################################################################################
x.loc[:,'ps_reg_03'] = pd.cut(x['ps_reg_03'], 50,labels=False)
x.loc[:,'ps_car_12'] = pd.cut(x['ps_car_12'], 50,labels=False)
x.loc[:,'ps_car_13'] = pd.cut(x['ps_car_13'], 50,labels=False)
x.loc[:,'ps_car_14'] =  pd.cut(x['ps_car_14'], 50,labels=False)
x.loc[:,'ps_car_15'] =  pd.cut(x['ps_car_15'], 50,labels=False)
#################################################################################

#################################################################################
test = x.iloc[train.shape[0]:].copy()
train = x.iloc[:train.shape[0]].copy()
logging.info('after pd cut, train shape : {0}'.format(train.shape))
logging.info('after pd cut, test shape : {0}'.format(test.shape))
#Always good to shuffle for SGD type optimizers
train = train.sample(frac=1, random_state=2017).reset_index(drop=True)
train.drop('id',inplace=True,axis=1)
test_id = test['id']
test.drop('id',inplace=True,axis=1)
#################################################################################

#################################################################################
categories = train.columns[1:]
numerics = []
currentcode = len(numerics)
catdict = {}
catcodes = {}
for x in numerics:
    catdict[x] = 0
for x in categories:
    catdict[x] = 1
#################################################################################

'''
#################################################################################
# prepare ffm train data
noofrows = train.shape[0]
noofcolumns = len(features)
# K folds
K = Number_of_folds
for fold_num, (train_idx, val_idx) in enumerate(comm_skf.split(train, train['target'])):
    logging.info('produce fold {0} train data'.format(fold_num))
    with open("train_ffm_{0}.dat".format(fold_num), "w") as f_train:
        for idx in train_idx:
            datastring = ""
            datarow = train.iloc[idx].to_dict()
            datastring += str(int(datarow['target']))
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
            datastring += '\n'
            f_train.write(datastring)
    logging.info('produce fold {0} valid data'.format(fold_num))
    with open("val_ffm_{0}.dat".format(fold_num), "w") as f_val:
        for idx in val_idx:
            datastring = ""
            datarow = train.iloc[idx].to_dict()
            datastring += str(int(datarow['target']))
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
            datastring += '\n'
            f_val.write(datastring)
    logging.info('fold {0} done'.format(fold_num))
#################################################################################


#################################################################################
# prepare ffm test data
noofrows = test.shape[0]
noofcolumns = len(features)
with open("all_test_ffm.dat", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if ((n % 100000) == 0):
            logging.info('Row {0}'.format(n))
        datastring = ""
        datarow = test.iloc[r].to_dict()
        datastring += str(int(datarow['target']))
        for i, x in enumerate(catdict.keys()):
            if (catdict[x] == 0):
                datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
            else:
                if (x not in catcodes):
                    catcodes[x] = {}
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode
                elif (datarow[x] not in catcodes[x]):
                    currentcode += 1
                    catcodes[x][datarow[x]] = currentcode
                code = catcodes[x][datarow[x]]
                datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"
        datastring += '\n'
        text_file.write(datastring)
#################################################################################
'''

'''
#################################################################################
ffm_train_bin = '/home/ubuntu/xbf/libffm/ffm-train'
ffm_predict_bin = '/home/ubuntu/xbf/libffm/ffm-predict'
all_test_dat = 'all_test_ffm.dat'
l_candidate = list(set(np.random.uniform(0.000005, 0.00002, 30)))
best_cv = 0.0
best_l = None
for l in l_candidate:
    logging.info('regularization value : {0}'.format(l))
    params = '-t 20 -r 0.1 -l {0}'.format(l)
    full_val_pred = np.zeros(np.shape(train)[0])
    full_test_pred = np.zeros(np.shape(test)[0])
    for f, (train_idx, val_idx) in enumerate(comm_skf.split(train, train['target'])):
        val_dat = 'val_ffm_{0}.dat'.format(f)
        train_dat = 'train_ffm_{0}.dat'.format(f)
        model = 'ffm_model_{0}'.format(f)
        output_test = 'ffm_output_test_{0}.dat'.format(f)
        output_val = 'ffm_output_val_{0}.dat'.format(f)
        logging.info('val_dat {0}, train_dat {1}, model {2}, output_test {3}, output_val {4}'.format(
            val_dat, train_dat, model, output_test, output_val))
        # train in fold
        logging.info(subprocess.call(' '.join([ffm_train_bin, '-p ' + val_dat, params, train_dat, model]), shell=True))
        # val in fold
        logging.info(subprocess.call(' '.join([ffm_predict_bin, val_dat, model, output_val]), shell=True))
        val_pred = pd.read_csv(output_val, header=None)
        val_pred.columns = ['target']
        full_val_pred[val_idx] = val_pred['target'].values
        logging.info('gini score in fold {0} : {1}'.format(
            f, gini_score_libffm(train.loc[val_idx]['target'], full_val_pred[val_idx])))
        # predict in fold
        logging.info(subprocess.call(' '.join([ffm_predict_bin, all_test_dat, model, output_test]), shell=True))
        test_pred = pd.read_csv(output_test, header=None)
        test_pred.columns = ['target']
        full_test_pred += test_pred.target.ravel() / Number_of_folds
        # clear all bin file
        logging.info(subprocess.call('rm *.bin', shell=True))
        gc.collect()

    cv_score = gini_score_libffm(train['target'].values, full_val_pred)
    if cv_score > best_cv:
        best_cv = cv_score
        best_l = l
    logging.info('cv score : {0}'.format(cv_score))

logging.info('best cv score {0}, best l : {1}'.format(best_cv, best_l))
#################################################################################
'''

# st(context=21)
ffm_train_bin = '/home/ubuntu/xbf/libffm/ffm-train'
ffm_predict_bin = '/home/ubuntu/xbf/libffm/ffm-predict'
all_test_dat = 'all_test_ffm.dat'
params = '-t 16 -r 0.1 -l 1.5312409909884344e-05'
full_val_pred = np.zeros(np.shape(train)[0])
full_test_pred = np.zeros(np.shape(test)[0])
for f, (train_idx, val_idx) in enumerate(comm_skf.split(train, train['target'])):
    val_dat = 'val_ffm_{0}.dat'.format(f)
    train_dat = 'train_ffm_{0}.dat'.format(f)
    model = 'ffm_model_{0}'.format(f)
    output_test = 'ffm_output_test_{0}.dat'.format(f)
    output_val = 'ffm_output_val_{0}.dat'.format(f)
    logging.info('val_dat {0}, train_dat {1}, model {2}, output_test {3}, output_val {4}'.format(
        val_dat, train_dat, model, output_test, output_val))
    # train in fold
    logging.info(subprocess.call(' '.join([ffm_train_bin, '-p ' + val_dat, params, train_dat, model]), shell=True))
    # val in fold
    logging.info(subprocess.call(' '.join([ffm_predict_bin, val_dat, model, output_val]), shell=True))
    val_pred = pd.read_csv(output_val, header=None)
    val_pred.columns = ['target']
    full_val_pred[val_idx] = val_pred['target'].values
    logging.info('gini score in fold {0} : {1}'.format(
        f, gini_score_libffm(train.loc[val_idx]['target'], full_val_pred[val_idx])))
    # predict in fold
    logging.info(subprocess.call(' '.join([ffm_predict_bin, all_test_dat, model, output_test]), shell=True))
    test_pred = pd.read_csv(output_test, header=None)
    test_pred.columns = ['target']
    full_test_pred += test_pred.target.ravel() / Number_of_folds
    # clear all bin file
    logging.info(subprocess.call('rm *.bin', shell=True))

cv_score = gini_score_libffm(train['target'].values, full_val_pred)
logging.info('cv score : {0}'.format(cv_score))

sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = full_test_pred
sub.to_csv('ffm_submission.csv',index=False)
#################################################################################


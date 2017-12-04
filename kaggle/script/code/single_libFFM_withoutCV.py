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



'''
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

noofrows = train.shape[0]
noofcolumns = len(features)
with open("alltrainffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if ((n % 100000) == 0):
            print('Row', n)
        datastring = ""
        datarow = train.iloc[r].to_dict()
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

noofrows = test.shape[0]
noofcolumns = len(features)
with open("alltestffm.txt", "w") as text_file:
    for n, r in enumerate(range(noofrows)):
        if ((n % 100000) == 0):
            print('Row', n)
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


#################################################################################
ffm_train_bin = '/home/ubuntu/xbf/libffm/ffm-train'
ffm_predict_bin = '/home/ubuntu/xbf/libffm/ffm-predict'
params = '-t 21'
train_data = 'alltrainffm.txt'
test_data = 'alltestffm.txt'
model = 'alltrainffm.txt.model'
output_avg = np.zeros(len(test_id))
AVG = 5
for i in range(AVG):
    output_data = 'output_{0}.txt'.format(i)
    logging.info(subprocess.call(' '.join([ffm_train_bin, train_data]), shell=True))
    logging.info(subprocess.call(' '.join([ffm_predict_bin, test_data, model, output_data]), shell=True))
    output = pd.read_csv(output_data, header=None)
    output.columns = ['target']
    output_avg += output.target.ravel() / AVG


sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = output_avg
sub.to_csv('ffm_submission.csv',index=False)
#################################################################################

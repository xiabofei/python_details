from flask import Flask, request
import json
from cervical_cancer import rating_engine
import cPickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import FACTOR_DESCRIPTION

app = Flask(__name__)

lr = cPickle.load(open('./data/support/trained_model_pkl/lr_2.pkl'))

levels_1 = {'P': [0, 3], 'Q': [0, 2], 'R': [0, 3], 'S': [0, 3], 'T': [0, 3], 'U': [0, 3], 'V': [0, 3], 'AE': [0, 4],
            'AF': [0, 4], 'AG': [0, 4], 'AH': [0, 3], 'IY': [1, 3]}

levels_2 = {"BD": [3.5, 9.5], "BE": [1.8, 6.3], "BF": [1.1, 3.2], "BG": [100, 160], "BH": [100, 300], "BI": [5, 49]}

@app.route('/risk-predict', methods=['GET'])
def risk_predict():
    factors = { k:float(v) for k,v in request.args.items() }
    return str(_lr_predict(factors))

# def _lr_predict(input_paras):
#     """
#     Use trained lr model to predict risk probability of cervical cancer recurrence
#     :input_paras input_paras: dict, {'DS':xxx, 'DK':yyy, ...}
#     :return: float, certain risk probability
#     """
#     X = pd.DataFrame(columns=lr['features'], dtype='float')
#     for f in lr['features']:
#         X[f] = [input_paras.get(f, 0)]
#     X = (X - lr['Xmin']) / (lr['Xmax'] - lr['Xmin']).replace(0, 1)
#     predict_results = dict(zip(lr['labels'], [r.predict_proba(X=X)[0,1] for r in lr['regressions']]))
#     return predict_results['IT']

def _lr_predict(input_paras):
    Xdict = dict()
    for key in input_paras.keys():
        value = float(input_paras[key])
        if key in levels_1.keys():
            for i in range(levels_1[key][0], levels_1[key][1] + 1):
                Xdict[key + "_" + str(i)] = 1.0 if value == i else 0.0
        elif key in levels_2.keys():
            for level in ['L', 'M', 'H']:
                Xdict[key + "_" + level] = 0.0
            if value < levels_2[key]:
                Xdict[key + "_L"] = 1.0
            elif value > levels_2[key]:
                Xdict[key + "_H"] = 1.0
            else:
                Xdict[key + "_M"] = 1.0
        else:
            Xdict[key] = value
    X = pd.DataFrame(columns=lr['features'], dtype='float')
    for f in lr['features']:
        X[f] = [Xdict.get(f, 0)]
    X = (X - lr['Xmin']) / (lr['Xmax'] - lr['Xmin']).replace(0, 1)
    r = lr['regressions']
    return r.predict_proba(X=X)[0, 1]

@app.route('/model-feature-weight', methods=['GET'])
def model_feature_weight():
    # from ipdb import set_trace as st
    # st(context=21)
    column_name = rating_engine._risk_factors
    ret = [ (k, {'name':column_name[k][FACTOR_DESCRIPTION],'weight':v }) for k,v in _lr_weight().items() ]
    return json.dumps(_sorting_feature_weights(ret))

def _sorting_feature_weights(feature_weights):
    """
    Sorting feature weights according to their abs values
    :input_paras feature_weights: list, [('column',{'name':xxx, 'weight':yyy}),... ]
    :return: sorted list
    """
    feature_weights.sort(key=lambda x:abs(x[1]['weight']), reverse=True)
    return feature_weights

def _lr_weight():
    """
    Return feature weight of Logistic Regression model
    :return: dict, {'DS':xxx, 'DK':yyy, ...}
    """
    feature_weights = dict(zip(lr['labels'], [dict(zip(lr['features'], r.coef_.squeeze().tolist())) for r in lr['regressions']]))
    return feature_weights['IT']

if __name__ == '__main__':
    app.run(port=5002)

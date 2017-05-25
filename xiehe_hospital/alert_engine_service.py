from flask import Flask, request
import json
from cervical_cancer import rating_engine
import cPickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

from config import FACTOR_DESCRIPTION

app = Flask(__name__)

lr = cPickle.load(open('./data/support/trained_model_pkl/lr_1.pkl'))

@app.route('/risk-predict', methods=['GET'])
def risk_predict():
    factors = { k:float(v) for k,v in request.args.items() }
    return str(_lr_predict(factors))

def _lr_predict(input_paras):
    """
    Use trained lr model to predict risk probability of cervical cancer recurrence
    :param input_paras: dict, {'DS':xxx, 'DK':yyy, ...}
    :return: float, certain risk probability
    """
    X = pd.DataFrame(columns=lr['features'], dtype='float')
    for f in lr['features']:
        X[f] = [input_paras.get(f, 0)]
    X = (X - lr['Xmin']) / (lr['Xmax'] - lr['Xmin']).replace(0, 1)
    predict_results = dict(zip(lr['labels'], [r.predict_proba(X=X)[0,1] for r in lr['regressions']]))
    return predict_results['IT']

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
    :param feature_weights: list, [('column',{'name':xxx, 'weight':yyy}),... ]
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

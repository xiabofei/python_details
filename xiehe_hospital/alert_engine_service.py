from flask import Flask, request
import cPickle
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/', methods=['GET'])
def risk_predict():
    factors = { k:float(v) for k,v in request.args.items() }
    return str(_lr_predict(factors))


lr = cPickle.load(open('./data/support/trained_model_pkl/lr.pkl'))
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
    outcome = dict(zip(lr['labels'], [r.predict_proba(X=X)[0,1] for r in lr['regressions']]))
    return outcome['IT']



if __name__ == '__main__':
    app.run(port=5002)

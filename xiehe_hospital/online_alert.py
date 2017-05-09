#encoding=utf8
from flask import Flask
from flask_bootstrap import Bootstrap
from flask import render_template, request

from cervical_cancer import rating_engine


from ipdb import set_trace as st

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    paras = {}
    # st(context=21)
    paras['factor_category_list'] = engine.produce_factor_category_list()
    engine.calculate_risk()
    return render_template('risk_predictor.html', **paras)


@app.route('/risk-predict', methods=['GET','POST'])
def risk_predict():
    # st(context=21)
    factors = request.args
    for k,v in factors.items():
        print k,v
    return str(engine.calculate_risk())


engine = rating_engine({}, 'IT', './data/support/lr.pkl')

if __name__ == '__main__':
    app.run()

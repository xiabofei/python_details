#encoding=utf8
from flask import Flask
from flask_bootstrap import Bootstrap
from flask import render_template, request
from random import randint
import requests
import json

from cervical_cancer import rating_engine


from ipdb import set_trace as st

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    paras = {}
    paras['factor_category_list'] = engine.produce_factor_category_list()
    return render_template('risk_predictor.html', **paras)


@app.route('/risk-predict', methods=['GET','POST'])
def risk_predict():
    factors = engine.transform_factors(request.args)
    risk_probability = float(requests.get('http://127.0.0.1:5002', params=factors).text)*100
    return "{0:.2f}".format(risk_probability)


engine = rating_engine({}, 'IT')

if __name__ == '__main__':
    app.run(port=5001)

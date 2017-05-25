#encoding=utf8
from flask import Flask
from flask_bootstrap import Bootstrap
from flask import render_template, request
from random import randint
import requests
import json

from cervical_cancer import rating_engine
from config import TARGET_RISK_FACTOR_RADAR, TARGET_RISK_FACTOR_POLAR


from ipdb import set_trace as st

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    paras = {}
    paras['factor_category_list'] = engine.produce_factor_category_list()
    paras['risk_factor_radar_data'] = engine.produce_risk_factor_radar_data()
    paras['feature_shown_in_polar'] = engine._clinical_target[engine.target][TARGET_RISK_FACTOR_POLAR]
    return render_template('risk_predictor.html', **paras)


@app.route('/risk-predict', methods=['GET','POST'])
def risk_predict():
    """
    如果risk predict service 部署在另一台机器上, 可能需要处理跨站请求保护的问题
    :return:
    """
    factors = engine.transform_factors(request.args)
    risk_probability = float(requests.get('http://127.0.0.1:5002/risk-predict', params=factors).text)*100
    return "{0:.2f}".format(risk_probability)

@app.route('/model-feature-weight', methods=['GET','POST'])
def model_feature_weight():
    """
    如果model feature weight service部署在另一台机器上, 可能需要处理跨站请求保护的问题
    :return:
    """
    features_weights = requests.get('http://127.0.0.1:5002/model-feature-weight').text
    return features_weights

engine = rating_engine({}, 'IT')

if __name__ == '__main__':
    app.run(port=5001)

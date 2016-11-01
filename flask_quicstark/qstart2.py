#encoding=utf8
from flask import Flask, url_for, render_template, abort, redirect, make_response

app = Flask(__name__)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login')
def login():
    abort(404)
    # this is never executed

@app.errorhandler(404)
def page_not_found(error):
    resp = make_response(render_template('error.html'),404)
    resp.headers['X-Something'] = 'A value'
    return resp

if __name__ == '__main__':
    app.run()

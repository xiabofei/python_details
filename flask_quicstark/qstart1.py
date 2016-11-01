#encoding=utf8
from flask import Flask, url_for, render_template

app = Flask(__name__)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)

@app.route('/')
def index():
    return 'defalut root response'

if __name__ == '__main__':
    app.run()

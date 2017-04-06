#encoding=utf8
from flask import Flask, jsonify, abort, make_response, request, url_for

app = Flask(__name__)

@app.route('/')
def index():
    return "ENCAPSULATE CCH API AS WEB SERVICE"

if __name__ == '__main__':
    app.run(debug=True)


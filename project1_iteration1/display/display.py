from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import json

app = Flask(__name__, template_folder='./templates')
bootstrap = Bootstrap(app)


@app.route('/')
def marker_incidents_on_map():
    """Marker incidents points on Google Map
    """
    paras = {}
    paras['incidents'] = json.load(open('../incidents_to_display.dat'))['incidents']
    return render_template('marker_incidents_on_map.html', **paras)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

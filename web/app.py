from flask import Flask, render_template
from flask_bootstrap import Bootstrap

app = Flask(__name__)
bootstrap = Bootstrap(app)


@app.route('/')
def index():
    title = 'owl monitor'
    return render_template('index.html', title=title)


@app.route('/webcam')
def webcam():
    title = 'owl monitor'
    return render_template('webcam.html', title=title)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

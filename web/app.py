import os
import sys
from flask import Flask, render_template,Response
from flask_bootstrap import Bootstrap

# -----------------------
# set path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)
os.system('')

# --------------------------------------------------------
# import /lib /pose
import src.test_webcam as test_webcam

# -----------------------
# flask stream
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


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(test_webcam.gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)

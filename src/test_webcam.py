import os
import sys
import cv2

# -----------------------
# set path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)
os.system('')

# --------------------------------------------------------
# import /lib /pose
from lib.web_camara import VideoCamera


def web_img():
    img_type = VideoCamera()
    return img_type


def gen():
    img_type = web_img()
    while True:
        frame = img_type.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame+ b'\r\n')


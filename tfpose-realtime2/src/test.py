import cv2
import math 
import os
import sys

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------
#setting
import lib.shared_setting as shared_setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")
DANGER_LABEL = config_all["danger_label"]
#---------------------------
#danger label
def is_danger_label(label):
  if label in set(DANGER_LABEL):
      print("D")
      
is_danger_label('walk')
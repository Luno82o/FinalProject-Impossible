import cv2
import math 
import os
import sys
import numpy as np

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------
#import lib
import lib.shared_setting as shared_setting

#---------------------------
#draw cv2 by 關鍵範圍+label

def draw_rect_label(img,id,skeleton,label):
    min_x=999
    min_y=999
    max_x=-999
    max_y=-999
    
    key=0
    
    while key < len(skeleton):
        if not (skeleton[key]== 0 or skeleton[key+1]== 0):
            min_x=min(min_x,skeleton[key])
            min_y=min(min_y,skeleton[key+1])
            max_x=max(max_x,skeleton[key])
            max_y=max(max_y,skeleton[key+1])
        key += 2
        
    min_x=int(min_x*img.shape[1])
    min_y=int(min_y*img.shape[0])
    max_x=int(max_x*img.shape[1])
    max_y=int(max_y*img.shape[0])
    
    #判斷危險標籤
    if shared_setting.is_danger_label(label):
        color=(0,0,255)
    else:
        color=(0,255,0)
        
    #畫出關鍵點方框
    img=cv2.rectangle(img,(min_x,min_y),(max_x,max_y),color, 4)
    
    #算出scale
    scale=max(0.5,min(2.0,(1.0*(max_x - min_x)/img.shape[1] / (0.3))**(0.5)))
    
    #標註label標籤
    org_x=int(min_x+5*scale)
    org_y=int(min_y-20*scale)
    fontFace=cv2.FONT_HERSHEY_SIMPLEX
    fontScale=1.4*scale
    linewidth=int(math.ceil(3*scale))
    
    img=cv2.putText(img, "#{}:{}".format(str(id%10),label), (org_x,org_y), fontFace, fontScale,color,linewidth,cv2.LINE_AA)
    
    
#draw score area
def score_area(img):
    r, c, d = img.shape
    blank = 0 + np.zeros((r, int(c/4), d), np.uint8)
    img = np.hstack((blank, img))
    return img
    
    
    
    
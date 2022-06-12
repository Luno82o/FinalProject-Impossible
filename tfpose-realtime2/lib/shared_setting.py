import yaml
import cv2
import os
import shutil
import simplejson
import pickle
import numpy as np
from os import listdir
from os.path import isfile, join, isdir

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
#--------------------------------
#path
def configPath(path):
    return ROOT + path

#----------------------------------------
#read yaml
def read_yaml(filepath):   
    with open(filepath, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded

#----------------------------------------
#save list
def save_list(filepath, ll):
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)

#----------------------------------------
#read list
def read_list(filepath):
    if os.path.isfile(filepath):        
        with open(filepath, 'r') as f:
            ll = simplejson.load(f)
    else: ll=[]
    return ll

#----------------------------------------
#save pickle
def save_model(filepath, model):
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

#----------------------------------------
#read pickle
def read_model(filepath):
    if os.path.isfile(filepath):
        with open(filepath, 'rb') as f:
            p=pickle.load(f)
    else: p=None
    return p
#----------------------------------------
#save csv
def save_csv(filepath,data,fmt="%i"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savetxt(filepath, data, fmt)

#----------------------------------------
#save csv
def read_csv(filepath,dtype=int):
    data=np.loadtxt(filepath, dtype)
    return data

#----------------------------------------
#delete data folder
def reset_data_folder(folder):
    if os.path.isdir(folder):
        shutil.rmtree(folder)
        os.mkdir(folder)
    else:
        os.mkdir(folder)
        
#--------------------------------
#get filenames
def get_filenames(path, use_sort=True, with_folder_path=False):
    
    fnames = [f for f in listdir(path) if isfile(join(path, f))]
    if use_sort:
        fnames.sort()
    if with_folder_path:
        fnames = [path + "/" + f for f in fnames]
        
    return fnames

#--------------------------------
#set avi
def set_avi(path,type_name,avi_name):
    if not isdir(path):
        os.makedirs(path)
    
    num=1
    n_path=path+type_name[0]+"_"+avi_name.format(num)
    while os.path.exists(n_path):
        num+=1
        n_path=path+type_name[0]+"_"+avi_name.format(num)
        
    return n_path

#---------------------------
#danger label

config_all = read_yaml(ROOT + "config/config.yaml")
DANGER_LABEL = config_all["danger_label"]


def is_danger_label(label):
    if label in set(DANGER_LABEL):
        return True
    return False

#--------------------------------
#Image Displayer    
class ImageDisplayer(object):
    def __init__(self):
        self._window_name = "video displayer"
        cv2.namedWindow(self._window_name)

    def display(self, image, wait_key_ms=1):
        cv2.imshow(self._window_name, image)
        cv2.waitKey(wait_key_ms)

    def __del__(self):
        cv2.destroyWindow(self._window_name)
        
#--------------------------------
#Image Writer
class ImageWriter(object):
    def __init__(self,path,fps):        
        self.path=path
        self.fps=fps
        
        self.current_img=0
        self.video_writer= None
            
    def write(self,img):
        self.current_img+=1
        if self.current_img==1:
            self.video_writer=cv2.VideoWriter(self.path,cv2.VideoWriter_fourcc(*'XVID')
                                              ,self.fps,(img.shape[1],img.shape[0]))        
        self.video_writer.write(img)
        
    def __del__(self):
        if self.current_img>0:
            self.video_writer.release()
            print("--------------------------")
            print("[已將照片寫入 /output]")
            print("   |FPS： {:>6} fps|".format(self.fps))
            print("   |影片秒數： {:>4} s|".format(self.current_img/self.fps))
            
        
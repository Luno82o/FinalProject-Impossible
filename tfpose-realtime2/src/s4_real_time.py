import sys
import os
import argparse
import cv2

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /lib /pose
import lib.shared_setting as shared_setting
import lib.img_type_setting as img_type_setting
import pose.pose_draw as pose_draw

from pose.pose_openpose import SkeletonDetector
from pose.pose_tracker import Tracker
from pose.pose_classifier import label_Live_Train

#--------------------------------
#choose type
#python s4_real_time.py -t video
def choose_type():
    parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
    parser.add_argument('-t','--type', default="webcam", choices=["video", "image", "webcam"])
    parser.add_argument('-i','--input', default="")
            
    args = parser.parse_args()     
    return args

#--------------------------------
#setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")

#constant
CONFIG_S4 = config_all["s4_real_time"]
AVI_FORMAT = config_all["avi_format"]
LABEL = config_all["label"]
WINDOW_SIZE = int(config_all["window_size"])

#openpose setting
OPENPOSE_MODEL =CONFIG_S4["openpose"]["model"]
OPENPOSE_IMG_SIZE =CONFIG_S4["openpose"]["img_size"]

#input model path
MODEL_PATH = shared_setting.configPath(CONFIG_S4["input"]["model_path"])

#args
args=choose_type()
TYPE=args.type
INPUTPATH=args.input

#output pqth
MOVIE_PATH = shared_setting.configPath(CONFIG_S4["output"]["movie_path"])
VIDEO_FPS = float(CONFIG_S4["output"]["video_fps"])
AVI_NAME = CONFIG_S4["output"]["base_avi_name"]
RESIZE_ROW = CONFIG_S4["output"]["resize_row"]

#public argument
label=None

#--------------------------------
#loaf type
def load_type(args_type,args_path):
    if args_type=="video":
        img_type=img_type_setting.videoType(args_path)
        
    elif args_type=="image":
        img_type=img_type_setting.imageType(args_path)
       
    elif args_type=="webcam":
        img_type=img_type_setting.webcamType()
        
    return img_type
    
#--------------------------------
#remove bad skeleton
def rm_skel(skeletons):
    reserve_skel=[]
    for skel in skeletons:
        s_x=skel[2:2+13*2:2]
        s_y=skel[3:2+13*2:2]
        
        len_skel=len([x for x in s_x if x!=0])
        len_leg=len([x for x in s_x[-6:] if x!=0])
        high=max(s_y)-min(s_y)
        
        if len_skel>=5 and len_leg>0 and high:
            reserve_skel.append(skel)
    return reserve_skel

#--------------------------------
#draw img
def draw_img(img,humans,muti_skeleton_id,skeleton_detector,labels,scale_h,multiple_label):
    row,col=img.shape[0:2]
    resize_col=int(col*RESIZE_ROW/row)
    
    #按比例縮放指定大小
    new_img=cv2.resize(img,dsize=(resize_col,RESIZE_ROW))
    copy_img=new_img.copy()
    #繪製關鍵點
    skeleton_detector.drawTransparent(new_img, humans)
    
    #調整關節透明度
    new_img=cv2.addWeighted(new_img, 0.4, copy_img, 0.6, 0)
    
    
    
    #繪製關節範圍線+label標籤
    if len(muti_skeleton_id):
        for id,label in labels.items():
            skeleton=muti_skeleton_id[id]
            skeleton[1::2]=skeleton[1::2]/scale_h #將y關鍵點縮放回原始大小
            pose_draw.draw_rect_label(new_img,id,skeleton,label)
        
    #繪製score area
    new_img=pose_draw.score_area(new_img)
    
    if len(muti_skeleton_id):
        #label score list
        m_label = multiple_label.get_classify('min')
        m_label.draw_score(new_img)
    
    
    return new_img
    
    
#--------------------------------
#multiple person label
class multi_label(object):
    def __init__(self,model_path,label_name):
        self.classifer_id={}
        
        self.classifer=lambda p_id: label_Live_Train(model_path,label_name,WINDOW_SIZE,p_id)

    def classify(self,muti_skeleton_id):
        
        #清除不可見的人
        old_id=set(self.classifer_id)
        cur_id=set(muti_skeleton_id)
        not_view=list(old_id-cur_id)
        
        for view in not_view:
            del self.classifer_id[view]
        
        predit_label={} #預測label
        
        for id,skeleton in muti_skeleton_id.items():
            if id not in self.classifer_id: #增加新的人
                self.classifer_id[id]=self.classifer(id)
                
            predit_label[id]=self.classifer_id[id].predict(skeleton)
            
        return predit_label
    
    def get_classify(self,id):
        if len(self.classifer_id) ==0:
            return None
        if id=='min':
            id=min(self.classifer_id.keys())
        return self.classifer_id[id]
        

#--------------------------------
#get skeleton
def get_skeleton():
    #Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    tracker = Tracker()
    
    #model
    multiple_label=multi_label(MODEL_PATH,LABEL)
    
    #load img type
    img_type=load_type(TYPE, INPUTPATH)
    displayer=shared_setting.ImageDisplayer()
    
    #writer video
    avi_path=shared_setting.set_avi(MOVIE_PATH,TYPE,AVI_NAME+AVI_FORMAT)
    video_writer=shared_setting.ImageWriter(avi_path,VIDEO_FPS)
    
    #read img
    while img_type.has_img():
        img=img_type.read_img()
        if img is None:
            break
        new_img=img.copy()
        
        #當按下enter結樹讀取照片
        if cv2.waitKey(1) == 13:
            img_type.stop()
            break
        
        #skeletons
        humans = skeleton_detector.detect(img)
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        skeletons=rm_skel(skeletons)
        
        #多人關節點
        muti_skeleton_id = tracker.track(skeletons)
                
              
        #識別每個人的label
        global label
        if len(muti_skeleton_id):
            label=multiple_label.classify(muti_skeleton_id)
            min_id=min(muti_skeleton_id.keys())
            print("      |預測結果為： {:>12}|".format(label[min_id]))
            
        #繪製
        new_img=draw_img(new_img,humans,muti_skeleton_id,skeleton_detector,label,scale_h,multiple_label)   
        
        
        #display and write
        displayer.display(new_img)
        video_writer.write(new_img)
    
#--------------------------------
#main
if __name__ == "__main__":
    get_skeleton()
    
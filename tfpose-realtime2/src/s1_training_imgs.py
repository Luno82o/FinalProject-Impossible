import cv2
import sys
import os
import collections
#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /pose /lib
from pose.pose_openpose import SkeletonDetector
from pose.pose_tracker import Tracker
from pose.pose_skeletons_io import ReadOriginTxt
#import pose.pose_commons as pose_commons

from lib.shared_setting import ImageDisplayer
import lib.shared_setting as shared_setting


#--------------------------------
#setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml") 

#constant
CONFOG_IMG_FORMAT = config_all["image_format"]
CONFOG_SKELETON_FORMAT = config_all["skeleton_format"]
CONFIG_S1 = config_all["s1_training_imgs"]
LABEL = config_all["label"]

#openpose statment
OPENPOSE_MODEL = CONFIG_S1["openpose"]["model"]
OPENPOSE_IMG_SIZE = CONFIG_S1["openpose"]["img_size"]

#input origin img
IMG_ORIGIN_TXT =shared_setting.configPath(CONFIG_S1["input"]["images_origin_txt"])
IMG_FOLDER =shared_setting.configPath(CONFIG_S1["input"]["images_folder"])

#output skeleton img
IMG_INFO_TXT =shared_setting.configPath(CONFIG_S1["output"]["images_information_txt"])
IMG_SKELETONS_TXT =shared_setting.configPath(CONFIG_S1["output"]["images_skeletons_eachTXT_folder"])
IMG_SKELETONS_FOLDER =shared_setting.configPath(CONFIG_S1["output"]["images_skeleton_folders"])
ALL_SKELETONS_TXT =shared_setting.configPath(CONFIG_S1["output"]["all_skeletons_txt"])

#--------------------------------
#get skeleton
#建立skeleton img and txt資料夾內容與image_information.txt
def get_skeleton():
    #Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    tracker = Tracker()
    #讀取orignial.txt並將每張照片資訊寫進image_information.txt
    images = ReadOriginTxt(img_folder = IMG_FOLDER,
                             valid_imgs_txt = IMG_ORIGIN_TXT,
                             img_filename_format = CONFOG_IMG_FORMAT)    
    images.save_images_info(filepath=IMG_INFO_TXT)
    displayer = ImageDisplayer()
    
    #重設建立skeleton img txt資料夾
    shared_setting.reset_data_folder(IMG_SKELETONS_TXT)
    shared_setting.reset_data_folder(IMG_SKELETONS_FOLDER)
    #os.makedirs(IMG_SKELETONS_TXT, exist_ok=True)
    #os.makedirs(IMG_SKELETONS_FOLDER, exist_ok=True)
    
    #為每張照片增加關節點
    total_images = images.num_images
    for cur_img in range(total_images):
        #當按下enter結樹讀取照片
        if cv2.waitKey(1) == 13:
            break
        img, label, info = images.read_image()
    
        humans = skeleton_detector.detect(img)
        
        #播放照片中的關節點
        img_c = img.copy()
        skeleton_detector.draw(img_c, humans)
        displayer.display(img_c, wait_key_ms=1)
    
        #記錄關節點座標
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        
        #存取多人關節點到個別txt img
        muti_skeleton = tracker.track(skeletons)
        skels_info = [info + skeleton.tolist()
                         for skeleton in muti_skeleton.values()]
        #txt       
        txtname = CONFOG_SKELETON_FORMAT.format(cur_img)
        print(txtname)
        shared_setting.save_list(IMG_SKELETONS_TXT+txtname , skels_info)
        #img
        imgname = CONFOG_IMG_FORMAT.format(cur_img)
        print(IMG_SKELETONS_FOLDER+imgname)
        cv2.imwrite(IMG_SKELETONS_FOLDER+imgname , img_c)

#--------------------------------
#get all skeleton txt
#從skeleton_txt 資料夾抓取每個txt資料寫入到skeletons_information.txt
def get_all_skeleton():
    print("----------------------------------------")
    print("(已將所有照片資訊寫進 image_information.txt)")
    print("----------------------------------------")
    print("(已合併txt 檔至 skeletons_information.txt)")
    filepaths = shared_setting.get_filenames(IMG_SKELETONS_TXT, with_folder_path=True)    
    all_skeletons = []
    labels_num = collections.defaultdict(int)
    
    #txt_n=0
    sum_txt=len(filepaths)
    for i in range(len(filepaths)): 
        #讀取每個skeletons txt
        filename = IMG_SKELETONS_TXT+CONFOG_SKELETON_FORMAT.format(i)
        skeletons= shared_setting.read_list(filename)
        
        if not skeletons:
            continue
        skeleton = skeletons[0]
        label = skeleton[3]
        if label not in LABEL:
            continue
        labels_num[label] += 1

        # Push to result
        all_skeletons.append(skeleton)

        
    #寫入skeletons_information.txt
    shared_setting.save_list(ALL_SKELETONS_TXT , all_skeletons)
        
    print("==>合計張數：{:>6}張".format(sum_txt))
    print("----------------------------------------")
    for label in LABEL:
        print("      |{:>12}--> {:>6}張|".format(label,labels_num[label]))
        
    
#--------------------------------
#main
if __name__ == "__main__":
    get_skeleton()   
    get_all_skeleton()
    
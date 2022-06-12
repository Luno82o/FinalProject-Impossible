import sys
import os
import numpy as np

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /lib /pose
import lib.shared_setting as shared_setting
import pose.pose_skeletons_io as ReadTxt
import pose.pose_feature as pose_feature

#--------------------------------
#setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")

#constant
CONFIG_S2 = config_all["s2_preprocess"]
LABEL = config_all["label"]
WINDOW_SIZE = int(config_all["window_size"])

#input skeletons folder path
ALL_SKELETONS_TXT =shared_setting.configPath(CONFIG_S2["input"]["all_skeletons_txt"])

#output all skeletons txt path
SKELETONS_FEATURES =shared_setting.configPath(CONFIG_S2["output"]["skeletons_features"])
SKELETONS_LABEL =shared_setting.configPath(CONFIG_S2["output"]["skeletons_label"])

#--------------------------------
#process features
def process_features(featureL, all_labelL, all_clip, classes):
    
    ADD_NOISE = False
    if ADD_NOISE:
        X1, Y1 = pose_feature.multi_frame_features(
            featureL, all_labelL, all_clip, WINDOW_SIZE, 
            is_adding_noise=True, is_print=True)
        X2, Y2 = pose_feature.multi_frame_features(
            featureL, all_labelL, all_clip, WINDOW_SIZE,
            is_adding_noise=False, is_print=True)
        X = np.vstack((X1, X2))
        Y = np.concatenate((Y1, Y2))
        return X, Y
    else:
        X, Y = pose_feature.multi_frame_features(
            featureL, all_labelL, all_clip, WINDOW_SIZE, 
            is_adding_noise=False, is_print=True)
        return X, Y
    
   
    
#--------------------------------
#preprocess
def preprocess():
    featureL,all_labelL,all_clip=ReadTxt.load_skeleton_data(ALL_SKELETONS_TXT,LABEL)
    
    print("----------------------------------------")
    print("[資料預處理]")
    feature, all_label=process_features(featureL, all_labelL, all_clip, LABEL)
    
    print("----------------------------------------")
    shared_setting.save_csv(SKELETONS_FEATURES, feature,"%.5f")
    shared_setting.save_csv(SKELETONS_LABEL, all_label)
    print("(已儲存 skeletons_features.csv , skeletons_label.csv)")

#--------------------------------
#main
if __name__ == "__main__":
    preprocess() 
import numpy as np
import cv2
import os
import simplejson
import sys

#-----------------------------------------------
#seting
LEN_IMG_INFO = 5
LEN_SKELETONS = 36

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)
import lib.shared_setting as shared_setting

config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")
CONFOG_IMG_FORMAT = config_all["image_format"]

#-----------------------------------------------
#Read ValidImages And Action Types By Txt
class ReadOriginTxt(object):
    
    #initialize
    def __init__(self, img_folder, valid_imgs_txt,
                 img_filename_format=CONFOG_IMG_FORMAT):
        self.images_info = get_training_imgs_info(
            valid_imgs_txt, img_filename_format)
        self.imgs_path = img_folder
        self.i = 0
        self.num_images = len(self.images_info)
        #print(f"原始照片位置: {img_folder}")
        #print(f"原始照片資訊記錄位置: {valid_imgs_txt}")
        #print(f"    Num images = {self.num_images}\n")
    
    #save images information 
    def save_images_info(self, filepath):
        folder_path = os.path.dirname(filepath)
        os.makedirs(folder_path, exist_ok=True)
        with open(filepath, 'w') as f:
            simplejson.dump(self.images_info, f)
            
    #get file name
    def get_filename(self, index):
        #jump_001/00001.jpg      
        return self.images_info[index-1][4]
    
    #get action label
    def get_action_label(self, index):
        #jump
        return self.images_info[index-1][3]

    #get image information
    def get_image_info(self, index):
        #[1, 7, 54, "jump", "jump_001/00001.jpg"]
        return self.images_info[index-1]
    
    #read images
    def read_image(self):
        self.i += 1
        if self.i > len(self.images_info):
            raise RuntimeError(f"There are only {len(self.images_info)} images, "
                               f"but you try to read the {self.i}th image")
        filepath = self.get_filename(self.i)
        img = self.imread(self.i)
        if img is None:
            raise RuntimeError("The image file doesn't exist: " + filepath)
        img_action_label = self.get_action_label(self.i)
        img_info = self.get_image_info(self.i)
        return img, img_action_label, img_info
    
    #read images in cv2
    def imread(self, index):
        return cv2.imread(self.imgs_path + self.get_filename(index))
    
    
#-----------------------------------------------
#get training imgs information
def get_training_imgs_info(valid_images_txt,img_filename_format=CONFOG_IMG_FORMAT):
    
    images_info = list()
    
    #open data txt
    with open(valid_images_txt) as f:
        
        folder_name = None
        action_label = None
        cnt_action = 0
        actions = set()
        action_images_cnt = dict()
        cnt_clip = 0
        cnt_image = 0
        
        #set sequence for all row e.g:(0,"abcd")
        for cnt_line, line in enumerate(f):
            
            #尋找txt中的資料夾名稱
            if line.find('_') != -1:
                folder_name = line[:-1]
                action_label = folder_name.split('_')[0]
                
                #紀錄新的動作標記
                if action_label not in actions:
                    cnt_action += 1
                    actions.add(action_label)
                    action_images_cnt[action_label] = 0
                    
            #當txt中有字串但不是資料夾名稱 存取數字值
            elif len(line) > 1:
                indices = [int(s) for s in line.split()]
                idx_start = indices[0]
                idx_end = indices[1]
                cnt_clip += 1
                for i in range(idx_start, idx_end+1):
                    filepath = folder_name+"/" + img_filename_format.format(i)
                    cnt_image += 1
                    action_images_cnt[action_label] += 1
                    
                    image_info = [cnt_action, cnt_clip,
                                  cnt_image, action_label, filepath]
                    assert(len(image_info) == LEN_IMG_INFO)
                    images_info.append(image_info)
                    # An example: [8, 49, 2687, 'wave', 'wave_03-02-12-35-10-194/00439.jpg']
                    # 累積動作數量, 目前停留的片段列, 累積訓練照片數量, 動作標記, 檔案名稱
                    
        print("---------------------------------------------------------")
        print("動作數量 = {}個".format(len(actions)))
        print("訓練照片數量 = {}張".format(cnt_image))
        print("每個動作訓練數量:")
        for action in actions:
            print("  |{:>12}--> {:>6}張|".format(
                action, action_images_cnt[action]))
        print("---------------------------------------------------------")
    return images_info
                    

#-----------------------------------------------
#load skeleton data

def load_skeleton_data(filepath, classes):
    
    label2index = {c: i for i, c in enumerate(classes)} #ex.{'jump' : 0,'wave' : 1}
    
    with open(filepath, 'r') as f:
        dataset = simplejson.load(f)
        
        #所有data
        dataset = [row for i, row in enumerate(dataset)]
        
        #關節點36個座標陣列
        feature = np.array([row[LEN_IMG_INFO:LEN_IMG_INFO+LEN_SKELETONS]for row in dataset])        
       
        #所有片段列
        all_clip = [row[1] for row in dataset]       
        
        #所有label
        all_label_str = [row[3] for row in dataset]
        all_label = [label2index[label] for label in all_label_str]#將label轉為數值
        
        #移除無效關節點
        if 0:
            vaild_sk=vaild_skeleton(dataset,feature)
            feature=feature[vaild_sk, :]
            all_label=[all_label[i] for i in vaild_sk]
            print("剩餘資料數：{:>5}筆".format(len(all_label)))
    
    return feature,all_label,all_clip
        
        
def vaild_skeleton(dataset,feature,NaN=0):
    print("-------------------------")
    print("丟棄無效或有缺失的關節點：")
    def is_valid(each_feature,i):
        is_v=len(np.where(each_feature[0:28] == NaN)[0]) == 0
        if(not is_v): print("  | delete {} |".format(dataset[i][4]))
        return is_v #若有0~14的關節點座標為NaN會捨棄
    
    vaild_sk = [i for i, each_data in enumerate(feature) if is_valid(each_data,i)]
    print("-------------------------")
    return vaild_sk
    
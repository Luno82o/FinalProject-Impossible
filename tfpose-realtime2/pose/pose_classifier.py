import os
import sys
import cv2
import numpy as np
from collections import deque
#----------------------------------
#import sklearn
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /lib /pose
import lib.shared_setting as shared_setting
from pose.pose_feature import FeatureGenerator

#----------------------------------
#setting
PCA_NUM=50
DEQUE_MAX_SIZE=2
PREDICT_MIN_ACCR=0.5


#----------------------------------
#label train
class labelTrain(object):
    def __init__(self):
        self.all_model()
        self.model=self.get_model("Neural_Net")
    
    def all_model(self):
        self.name=["Neural_Net"]
        self.classifer=[MLPClassifier(hidden_layer_sizes=(20, 30, 40))]
        
    def get_model(self,name):
        index=self.name.index(name)
        return self.classifer[index]
        
    def train(self, features, label):
        print("   |PCA")
        print("   |訓練前大小:  ({:>6},{:>3})|".format(*features.shape))
        #PCA
        if(len(label)<=PCA_NUM): 
            n_components=len(label)
        else:
            n_components = min(PCA_NUM, features.shape[1]) #保留PCA的數量
        self.pca= PCA(n_components, whiten=True) #whiten 讓每個特徵有相同特徵差
        a=self.pca.fit(features) #訊練PCA
        n_features = self.pca.transform(features) #降features維度
        print("   |訓練後大小:  ({:>6},{:>3})|".format(*n_features.shape)) 
        
        #get model
        self.model.fit(n_features, label)
               
    def predict(self, t_features, t_label):
        t_label_predict=self.model.predict(self.pca.transform(t_features))
        correct=sum(t_label_predict==t_label)
        accuracy=correct/len(t_label)
        
        return accuracy,t_label_predict
    
    def _predict_proba(self,features):
        label_predict=self.model.predict_proba(self.pca.transform(features))
        return label_predict
        
#label real time train
class label_Live_Train(object):
    
    def __init__(self,model_path,label, window_size, p_id=0):
        #load model
        self.model=shared_setting.read_model(model_path)
        if not self.model:
            print("-------------------------------")
            print("[model檔案連結錯誤！]")
            os._exit(0)
        print("-------------------------------")
        print("[讀取model成功]")
        
        self.label=label
        self.p_id=p_id
        
        self.feature_generator= FeatureGenerator(window_size)
        self.reset()
               
    def reset(self):
        self.feature_generator.reset()
        self.scores=None
        self.pre_scores= deque()
        
    def predict(self,skeleton):
        # Get features
        success, features = self.feature_generator.add_cur_skeleton(skeleton)
        
        if success:
            features = features.reshape(-1, features.shape[0]) #轉成二維
            cur_scores = self.model._predict_proba(features)[0]
            self.scores=self.smooth_scores(cur_scores)
            
            if self.scores.max()<PREDICT_MIN_ACCR:
                predict_label=""
            else:
                predict_id=self.scores.argmax()
                predict_label=self.label[predict_id]
            
        else:
            predict_label=""
            
        return predict_label
    
    def smooth_scores(self,cur_scores,type_calculate=1):
        #紀錄先前的scores
        self.pre_scores.append(cur_scores)        
        if len(self.pre_scores) > DEQUE_MAX_SIZE:
            self.pre_scores.popleft() #刪除最之前的紀錄
        
        #calculate type scores
        if type_calculate: #default id sum
            s_sum=np.zeros(len(self.label))
            for score in self.pre_scores:
                s_sum+=score
            s_sum/=len(self.pre_scores)
            return s_sum
        
        else:
            s_mul=np.ones(len(self.label))
            for score in self.pre_scores:
                s_mul*=score
            return s_mul
    
    #label score list
    def draw_score(self,img):
        fontFace=cv2.FONT_HERSHEY_SIMPLEX
        fontScale=0.7
        
        if self.scores is None:
            return
        
        max_score=max(self.scores)
        label_name=None
        color=None
        for i in range(-1,len(self.label)):
            
            if i==-1:
                text = "#{}:".format(str(self.p_id%10))             
            else:
                label_name=self.label[i]
                text = "{:<5}: {:.2f}".format(label_name, self.scores[i])
                
            if max_score==self.scores[i] and shared_setting.is_danger_label(label_name):
                color=(0,0,255)
            elif max_score==self.scores[i]: color=(0,255,0)
            else: color=(255,255,255)
            
            cv2.putText(img,text,(20,150+i*30),fontFace,fontScale,color,2)
            
        
    
        
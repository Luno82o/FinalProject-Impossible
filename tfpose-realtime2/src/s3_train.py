import sys
import os
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#--------------------------------------------------------
#set path
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

#--------------------------------------------------------
#import /lib /pose
import lib.shared_setting as shared_setting
from pose.pose_classifier import labelTrain
import pose.pose_draw as pose_draw


#--------------------------------
#setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")

#constant
CONFIG_S3 = config_all["s3_train"]
LABEL = config_all["label"]
LABEL_N = np.array(LABEL)

#input csv data path
SKELETONS_FEATURES =shared_setting.configPath(CONFIG_S3["input"]["skeletons_features"])
SKELETONS_LABEL =shared_setting.configPath(CONFIG_S3["input"]["skeletons_label"])

#output model path
MODEL_PATH = shared_setting.configPath(CONFIG_S3["output"]["model_path"])


#--------------------------------
#predict accuracy
def predict(model,label,tr_Features,te_Features,tr_Label,te_Label):    
    #train predict
    train_accu,train_pred=model.predict(tr_Features, tr_Label)
    print("   |訓練集：{:10.2f}%|".format(train_accu*100))
    
    #test predict
    test_accu,test_pred=model.predict(te_Features, te_Label)
    print("   |測試集：{:10.2f}%|".format(test_accu*100))

    #report
    print("---------------------------------")
    print("[模型報告]")
    report=classification_report(te_Label, test_pred, target_names=label, output_dict=False)   
    print(report)
    
    #plot
    print("---------------------------------")
    print("[混淆矩陣格式]")
    axis, cf = pose_draw.draw_confusion(te_Label,test_pred,label,
                           normalize=False,size=(4, 4))
    
    plt.show()
    
#--------------------------------
#train and predict model
def main():
    features=shared_setting.read_csv(SKELETONS_FEATURES,float)
    label=shared_setting.read_csv(SKELETONS_LABEL)
    
    #split train test
    #train 80%,test 20% and set randrom seed    #test_size=0.3 random_state=1
    train_Features,test_Features,train_Label,test_Label = train_test_split(features,label,test_size=0.2,random_state=777)
    print("---------------------------------")
    print("   |訓練大小:  ({:>6},{:>3})|".format(*train_Features.shape))
    print("   |訓練數量:   {:>9}筆|".format(len(train_Label)))
    print("   |測試數量:   {:>9}筆|".format(len(test_Label)))
    
    #train
    print("---------------------------------")    
    print("[訓練模型]")   
    model = labelTrain()
    model.train(train_Features, train_Label)
    
    #predit and report
    print("---------------------------------")
    print("[預測模型]")
    print("   |準確率")
    predict(model,LABEL_N,train_Features,test_Features,train_Label,test_Label)
    
    #save
    print("---------------------------------")
    print("(已儲存模型 model.pickle)")
    shared_setting.save_model(MODEL_PATH,model)
    



#--------------------------------
#main
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
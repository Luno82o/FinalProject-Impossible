import numpy as np
import cv2
import sys
import os
import argparse
import itertools

from shared_setting import ImageDisplayer
import shared_setting 
#--------------------------------
#setting
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"

config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")
CONFOG_IMG_FORMAT = config_all["image_format"]
CONFIG_ORIGN = config_all["orign_path"]
OUPUT_PATH =shared_setting.configPath(CONFIG_ORIGN["output"]["path"])

#--------------------------------
#parse args
def parse():
    parser = argparse.ArgumentParser(description='video to jpg')
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-v", "--interval", type=int, required=False,default=1)#nth frame
    parser.add_argument("-m", "--max", type=int, required=False,default=100000)#max frame
    args = parser.parse_args()
    return args
  
#--------------------------------
#parse args
def saveDir():
    global OUPUT_PATH
    repeat_num=1
    while True:
        if repeat_num>100:
            break;
        outputDir=OUPUT_PATH+"/Result_"+"{:03d}".format(repeat_num)
        if os.path.isdir(outputDir):
            repeat_num+=1
        else:
            os.makedirs(outputDir)
            break;
    OUPUT_PATH=outputDir

#--------------------------------
#set output filename
def saveJpg(i):
    return OUPUT_PATH + "/" + CONFOG_IMG_FORMAT.format(i)

#--------------------------------
#Read Video
class ReadVideo(object):
    def __init__(self, path, interval=1):
        
        #判斷是否有該資料
        if not os.path.exists(path):
            raise IOError("Video not exist: " + path)
          
        #VideoCapture
        self.video = cv2.VideoCapture(path)
        ret, frame = self.video.read()
        
        self.cnt_imgs = 0
        self.next_image = frame
        self.interval = interval
    
    def read_image(self):
        image = self.next_image
        for i in range(self.interval):
            if self.video.isOpened():
                ret, frame = self.video.read()
                self.next_image = frame
            else:
                self.next_image = None
                break
            self.cnt_imgs += 1
        return image
       
       
#--------------------------------
#main
def main(args):
    
    #讀取影片
    video_loader = ReadVideo(args.input)
    
    #建立OUPUT_PATH 資料夾
    saveDir()
    
    img_displayer = ImageDisplayer()
    current_img = 0
    
    for i in itertools.count():
        img = video_loader.read_image()
        if img is None:
            print("-----影片轉換照片完成----")
            break
        if i % args.interval == 0: #判斷為該frame
            current_img += 1
            print("Processing {}th image".format(current_img))
            cv2.imwrite(saveJpg(current_img), img) #jpg寫入file
            img_displayer.display(img)
            if current_img == args.max:
                print("Read {} frames. ".format(current_img) +
                      "Reach the max_frames setting. Stop.")
                break
            
    
#進入點
if __name__ == "__main__":
    args = parse()
    main(args)
    
  
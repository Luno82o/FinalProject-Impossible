# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 00:13:04 2022
"""

import cv2
import threading
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image    # 載入 PIL 相關函式庫

import tmp

text = ""

def videoCapture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    #cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE); # 命名一個視窗，可不寫
    while(True):
        # 擷取影像
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        fontpath = 'textfount/NotoSansTC-Regular.otf'          # 設定字型路徑
        font = ImageFont.truetype(fontpath, 50)      # 設定字型與文字大小
        imgPil = Image.fromarray(frame)                # 將 img 轉換成 PIL 影像
        draw = ImageDraw.Draw(imgPil)                # 準備開始畫畫
        draw.text((0, 0), text, fill=(0, 0, 0), font=font)  # 畫入文字，\n 表示換行
        frame = np.array(imgPil)                       # 將 PIL 影像轉換成 numpy 陣列
        
        
        # 顯示圖片
        cv2.imshow('live', frame)
        
        # 按下 q 鍵離開迴圈
        if cv2.waitKey(1) == ord('q'):
            break
        
    # 釋放該攝影機裝置
    cap.release()
    cv2.destroyAllWindows()
    return

    
def recognition():
    while (True):
        audio = tmp.recordStatement()
        
        global text
        text = tmp.recognizeCommand(audio)
        
        print(f"{text}\n")
        
        if "救命" in text:
            print("有人呼救")
    return


if __name__ == '__main__':
    
    t1 = threading.Thread(target = videoCapture)
    t2 = threading.Thread(target = recognition)
    
    t1.start()
    t2.start()
    
    t1.join()
    os._exit(0)

    
    

import argparse
import time
import cv2 as cv
import numpy as np

import lib.camara_setting as caset
import lib.model_setting as moset

from pose.pose_visualizer import TfPoseVisualizer
from lib.recognizer import ID

#------------------------------------------------------------------------
#default setting
parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
parser.add_argument('--video', help='Path to video file.') #webcam or video
args = parser.parse_args()

realtime_fps = 0
start_time = time.time()
fps_interval = 1
fps_count = 0
run_timer = 0
frame_count = 0

is_clear= False


#------------------------------------------------------------------------
#read and write video/webcam from camara_setting
cap = caset.choose_run_mode(args)
video_w = caset.set_video_writer(cap)  #video_w = caset.set_video_writer(cap, write_fps=int(7.0))

#load model
estimator = moset.load_pretrain_model('cmu')

#------------------------------------------------------------------------
#calculate FPS
def FPS():
    
    global start_time,fps_count,fps_count,realtime_fps
    
    fps_count += 1
    elapse_time= time.time()-start_time 
    
    if (elapse_time) > fps_interval:
        realtime_fps = fps_count / elapse_time
        fps_count=0
        start_time = time.time()
    
    fps_label = 'FPS:{0:.2f}'.format(realtime_fps)
    
    return fps_label
    


#------------------------------------------------------------------------
#total time and frame
def total_time_frame():
    
    global frame_count,run_timer
    
    frame_count += 1
    
    if frame_count == 1:
        run_timer = time.time()
    run_time = time.time() - run_timer
    time_frame_label = '[Time:{0:.2f} | Frame:{1}]'.format(run_time, frame_count)
    
    return time_frame_label,run_time



#------------------------------------------------------------------------
#number of people
def num_people(humans):
    num_label = "Human: {0}".format(len(humans))
    
    return num_label


#------------------------------------------------------------------------
#main (press enter exit)
while cv.waitKey(1) != 13:
    is_ret, frame = cap.read()
    if not is_ret:
        print("not receive frame, execute will exit.")
        break
    else:
        print("execute video stream.")
        
        #show FPS
        height, width = frame.shape[:2]
        cv.putText(frame, FPS(), (width-85, 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
       
        #show total time and frame
        cv.putText(frame, total_time_frame()[0], (5, height-15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        #human tfpose base model
        humans = estimator.inference(frame)
        pose = TfPoseVisualizer.draw_pose_rgb(frame, humans)
        
        
        #show people count
        num_label = "Human: {0}".format(len(humans))
        cv.putText(frame,num_label, (5, height-45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        #record point
        if(not is_clear):
            caset.clear_txt()
            is_clear=True
        
        caset.point_txt(pose,ID(pose),total_time_frame()[1])
        
        #save img in each frame
        caset.set_video_IMG(total_time_frame()[1],frame)
        
        cv.imshow('based on OpenPose', frame)
        video_w.write(frame)
        
    
    
video_w.release()
cap.release()
cv.destroyAllWindows()
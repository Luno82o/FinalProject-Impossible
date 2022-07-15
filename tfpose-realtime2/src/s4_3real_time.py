import sys
import os
import argparse
import cv2
import keyboard
# import threading
import numpy as np
import time
import copy
from PIL import ImageFont, ImageDraw, Image

# --------------------------------------------------------
# set path
ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"
sys.path.append(ROOT)
os.system('')

# --------------------------------------------------------
# import /lib /pose
import lib.shared_setting as shared_setting
import lib.img_type_setting as img_type_setting
import pose.pose_draw as pose_draw
import voice.voice_speech as voice_speech
import voice.voice_sound as voice_sound
# import voice.voice_sound_prediction as voice_sound_prediction

from pose.pose_openpose import SkeletonDetector
from pose.pose_tracker import Tracker
from pose.pose_classifier import label_Live_Train


# --------------------------------
# choose type
# python s4_real_time.py -t video
def choose_type():
    parser = argparse.ArgumentParser(description='Action Recognition by OpenPose')
    parser.add_argument('-t', '--type', default="webcam", choices=["video", "image", "webcam"])
    parser.add_argument('-i', '--input', default="")

    args = parser.parse_args()
    return args


# --------------------------------
# setting
config_all = shared_setting.read_yaml(ROOT + "config/config.yaml")

# constant
CONFIG_S4 = config_all["s4_real_time"]
AVI_FORMAT = config_all["avi_format"]
LABEL = config_all["label"]
WINDOW_SIZE = int(config_all["window_size"])

# openpose setting
OPENPOSE_MODEL = CONFIG_S4["openpose"]["model"]
OPENPOSE_IMG_SIZE = CONFIG_S4["openpose"]["img_size"]

# input model path
MODEL_PATH = shared_setting.configPath(CONFIG_S4["input"]["model_path"])
SOUND_MODEL_PATH = shared_setting.configPath(CONFIG_S4["input"]["sound_model_path"])
DANGER_DICT_PATH = shared_setting.configPath(CONFIG_S4["input"]["danger_dict_path"])

# args
args = choose_type()
TYPE = args.type
INPUTPATH = args.input

# output pqth
MOVIE_PATH = shared_setting.configPath(CONFIG_S4["output"]["movie_path"])
SOUND_PATH = shared_setting.configPath(CONFIG_S4["output"]["sound_path"])
SOUND_PATH_TMP = shared_setting.configPath(CONFIG_S4["output"]["sound_path_tmp"])

VIDEO_FPS = float(CONFIG_S4["output"]["video_fps"])
AVI_NAME = CONFIG_S4["output"]["base_avi_name"]
RESIZE_ROW = CONFIG_S4["output"]["resize_row"]
DANGER_MAX_NUM = 10
DANGER_TIMES_SEC = 2.0
DANGER_TIMES_ACC = 10.0

##public argument
label = None
# sound
speech_text = "not speech"
pre_speech_text = None
is_sound_danger = False

# img 
danger_report_dict = {}
is_danger_record = False
first_time_acc = None
first_time_sec = None
lock_first_danger = False
first_danger_label = None


# --------------------------------
# loaf type
def load_type(args_type, args_path):
    if args_type == "video":
        img_type = img_type_setting.videoType(args_path)

    elif args_type == "image":
        img_type = img_type_setting.imageType(args_path)

    elif args_type == "webcam":
        img_type = img_type_setting.webcamType()

    return img_type


# --------------------------------
# remove bad skeleton
def rm_skel(skeletons):
    reserve_skel = []
    for skel in skeletons:
        s_x = skel[2:2 + 13 * 2:2]
        s_y = skel[3:2 + 13 * 2:2]

        len_skel = len([x for x in s_x if x != 0])
        len_leg = len([x for x in s_x[-6:] if x != 0])
        high = max(s_y) - min(s_y)

        if len_skel >= 5 and len_leg > 0 and high:
            reserve_skel.append(skel)
    return reserve_skel


# --------------------------------
# draw img
def draw_img(img, humans, muti_skeleton_id, skeleton_detector, labels, scale_h, multiple_label):
    row, col = img.shape[0:2]
    resize_col = int(col * RESIZE_ROW / row)

    # 按比例縮放指定大小
    new_img = cv2.resize(img, dsize=(resize_col, RESIZE_ROW))
    copy_img = new_img.copy()
    # 繪製關鍵點
    skeleton_detector.drawTransparent(new_img, humans)

    # 調整關節透明度
    new_img = cv2.addWeighted(new_img, 0.4, copy_img, 0.6, 0)

    # 繪製關節範圍線+label標籤
    if len(muti_skeleton_id):
        for id, label in labels.items():
            skeleton = muti_skeleton_id[id]
            skeleton[1::2] = skeleton[1::2] / scale_h  # 將y關鍵點縮放回原始大小
            pose_draw.draw_rect_label(new_img, id, skeleton, label)

    # 繪製score area
    '''
    new_img=score_area(new_img,muti_skeleton_id,multiple_label,"max")
    '''

    return new_img


# --------------------------------
# score area
def score_area(new_img, muti_skeleton_id, multiple_label, type_n):
    new_img = pose_draw.score_area(new_img)

    if len(muti_skeleton_id):
        # label score list
        m_label = multiple_label.get_classify(type_n)
        m_label.draw_score(new_img)
    return new_img


# --------------------------------
# multiple person label
class multi_label(object):
    def __init__(self, model_path, label_name):
        self.classifer_id = {}

        self.classifer = lambda p_id: label_Live_Train(model_path, label_name, WINDOW_SIZE, p_id)

    def classify(self, muti_skeleton_id):

        # 清除不可見的人
        old_id = set(self.classifer_id)
        cur_id = set(muti_skeleton_id)
        not_view = list(old_id - cur_id)

        for view in not_view:
            del self.classifer_id[view]

        predit_label = {}  # 預測label

        for id, skeleton in muti_skeleton_id.items():
            if id not in self.classifer_id:  # 增加新的人
                self.classifer_id[id] = self.classifer(id)

            predit_label[id] = self.classifer_id[id].predict(skeleton)

        return predit_label

    def get_classify(self, id):
        if len(self.classifer_id) == 0:
            return None
        if id == 'min':
            id = min(self.classifer_id.keys())
        if id == 'max':
            id = max(self.classifer_id.keys())
        return self.classifer_id[id]


# --------------------------------
# information danger,用於label是危險狀態時   
def info_danger(id, label, all_id):
    global danger_report_dict
    data = {}

    # 若字典中的id(key)沒有在該幀中會被刪除
    diff = set(danger_report_dict).difference(set(all_id))
    [danger_report_dict.pop(key) for key in diff]

    # 沒有該id則新增該id的資訊
    if not danger_report_dict.__contains__(id):
        data["cur_label"] = label
        if shared_setting.is_danger_label(label):  # 若新的id是危險label則新增
            data[label] = 1
        danger_report_dict[id] = data

    # 有該id,開始記數    
    elif shared_setting.is_danger_label(label):
        danger_report_dict[id]["cur_label"] = label
        if danger_report_dict[id].__contains__(label):  # 已有該label,+1
            danger_report_dict[id][label] += 1
        else:  # 新增該label
            danger_report_dict[id][label] = 1
    else:
        danger_report_dict[id]["cur_label"] = label
    print(danger_report_dict)


# --------------------------------
# report danger
def report_danger(id):
    global danger_report_dict, is_danger_record
    global first_time_acc
    global lock_first_danger, first_time_sec, first_danger_label

    ##計算危險動作累積次數
    tmp_dict = copy.deepcopy(danger_report_dict)
    tmp_dict[id].pop("cur_label")
    max_danger_num = max(tmp_dict[id].values(), default=0)
    if not max_danger_num:
        first_time_acc = time.time()

    if max_danger_num >= DANGER_MAX_NUM and time.time() - first_time_acc <= DANGER_TIMES_ACC:
        first_time_acc = time.time()
        is_danger_record = True
        print("累積判定是危險")

    elif time.time() - first_time_acc > DANGER_TIMES_ACC:
        first_time_acc = time.time()
        # 要清除字典
        for key in danger_report_dict.keys():
            [danger_report_dict[key].pop(a) for a in copy.deepcopy(danger_report_dict)[key].keys() \
             if a != "cur_label"]
        print("超時1")

    ##連續危險動作>指定秒數
    tmp2_dict = copy.deepcopy(danger_report_dict)

    # 當 讀出危險label 鎖住目前狀態
    if shared_setting.is_danger_label(tmp2_dict[id]["cur_label"]) and not lock_first_danger:
        first_danger_label = tmp2_dict[id]["cur_label"]
        first_time_sec = time.time()
        lock_first_danger = True
    # 當 目前label跟第一次一樣 且 是鎖住的狀態 且 時間大於指定秒數
    elif tmp2_dict[id]["cur_label"] == first_danger_label and \
            lock_first_danger and time.time() - first_time_sec >= DANGER_TIMES_SEC:
        first_danger_label = None
        first_time_sec = None
        lock_first_danger = False
        is_danger_record = True
        print("秒數判定是危險")
    # 當 是鎖住的狀態 且 在指定秒數範圍都沒發生相同危險
    elif lock_first_danger and time.time() - first_time_sec >= DANGER_TIMES_SEC and \
            time.time() - first_time_sec <= DANGER_TIMES_SEC + 1.5:
        first_danger_label = None
        first_time_sec = None
        lock_first_danger = False
        print("超時2")


# --------------------------------
# get skeleton
def get_skeleton():
    global is_danger_record

    # Detector
    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)
    tracker = Tracker()

    # model
    multiple_label = multi_label(MODEL_PATH, LABEL)

    # load img type
    img_type = load_type(TYPE, INPUTPATH)
    displayer = shared_setting.ImageDisplayer()

    # read img
    while img_type.has_img():
        img = img_type.read_img()
        if img is None:
            break
        new_img = img.copy()

        # 當按下enter結束讀取照片
        if is_danger_record:
            img_type.stop()
            break

        if keyboard.is_pressed("e"):
            os._exit(0)

        # skeletons
        humans = skeleton_detector.detect(img)
        skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
        skeletons = rm_skel(skeletons)

        # 多人關節點
        muti_skeleton_id = tracker.track(skeletons)

        # 識別每個人的label
        global label
        if len(muti_skeleton_id):
            label = multiple_label.classify(muti_skeleton_id)
            for id, lab in label.items():
                info_danger(id, lab, label.keys())
                report_danger(id)
                if shared_setting.is_danger_label(lab):
                    print("      |#{}預測結果為： {:>12}|".format(str(id), "\033[31m" + lab + "\033[37m"))
                else:
                    print("      |#{}預測結果為： {:>12}|".format(str(id), "\033[32m" + lab + "\033[37m"))

        # 繪製
        new_img = draw_img(new_img, humans, muti_skeleton_id, skeleton_detector, label, scale_h, multiple_label)

        # display and write
        displayer.display(new_img)


# --------------------------------
# get skeleton
def recording_danger():
    global is_danger_record

    # load img type
    img_type = load_type(TYPE, INPUTPATH)
    displayer = shared_setting.ImageDisplayer(2)

    # writer video
    avi_path = shared_setting.set_avi(MOVIE_PATH, TYPE, AVI_NAME + AVI_FORMAT)
    video_writer = shared_setting.ImageWriter(avi_path, img_type.FPS())

    print("----錄製中-----")
    while img_type.has_img():
        img = img_type.read_img()
        if img is None:
            break
        new_img = img.copy()

        if keyboard.is_pressed("q"):
            print("----結束錄製---")
            is_danger_record = False
            img_type.stop()
            break

        # 增加時間
        cur_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

        fontpath = ROOT + 'lib/textfount/NotoSansTC-Regular.otf'
        font = ImageFont.truetype(fontpath, 25)
        imgPil = Image.fromarray(new_img)
        draw = ImageDraw.Draw(imgPil)
        draw.text((10, 10), cur_time, fill=(0, 0, 255), font=font)
        draw.text((10, 35), "錄影中", fill=(0, 255, 0), font=font)
        new_img = np.array(imgPil)

        # display and write
        displayer.display(new_img)
        video_writer.write(new_img)


# --------------------------------
# voice(語音辨識)
def voicespeech(filename):
    global speech_text, pre_speech_text
    pre_speech_text = speech_text

    audio, speech_text = voice_speech.recognition(filename)

    if (speech_text == "None"):
        speech_text = pre_speech_text
    else:
        pre_speech_text = speech_text
    print("===>語音辨識：" + speech_text)


# voice(聲音辨識)
def voicesound(filename):
    global is_danger_record
    filelist = voice_sound.cutwav(filename)
    for file in filelist:
        print(file)
        data = voice_sound.getmel(file)

        sound_predictions = voice_sound.getPridict(data, SOUND_MODEL_PATH)
        is_danger_record = voice_sound.is_danger_sound(sound_predictions)
        print("===>危險聲音：" + str(is_danger_record))

    # voice


# 偵測 儲存聲音        
def detectesound():
    recorder = voice_sound.Recorder(SOUND_PATH_TMP)
    print("================================S=T=A=R=T=================================")
    silence_start_time = recorder.getcurrenttime()
    print("------------------recording silence sound------------------")
    rec_silence = []

    while (True):
        data = recorder.getsound()
        if recorder.ifnoise(data):
            if rec_silence != '':
                silencefile = recorder.write(b''.join(rec_silence), silence_start_time)
            print("------------------stop recording silence sound------------------")
            rec_silence = []
            print("\n------------------recording noise sound------------------")
            rec_noise = []
            rec_noise.append(data)

            noise_start_time = recorder.getcurrenttime()

            currentime = time.time()
            endtime = time.time() + 2
            addtime = endtime
            while (currentime <= endtime) or (currentime <= addtime):
                data = recorder.getsound()
                if (addtime - 0.5 < currentime) and (currentime <= addtime):
                    if recorder.ifnoise(data):
                        print('test')
                        # addtime = addtime + 0.5
                rec_noise.append(data)

                currentime = time.time()

                if (is_danger_record):
                    noisefile = recorder.write(b''.join(rec_noise), noise_start_time)
                    recorder.merge_files(SOUND_PATH_TMP, SOUND_PATH)
                    return

            noisefile = recorder.write(b''.join(rec_noise), noise_start_time)
            print("------------------stop recording noise sound------------------")

            voicespeech(noisefile)
            voicesound(noisefile)

            print("\n------------------recording silence sound------------------")
            """
            t1 = threading.Thread(target = voicespeech(noisefile))
            t2 = threading.Thread(target = voicesound(noisefile))
            
            t1.start()
            t2.start()
            """

            silence_start_time = recorder.getcurrenttime()

        else:
            rec_silence.append(data)

            if (is_danger_record):
                silencefile = recorder.write(b''.join(rec_silence), silence_start_time)
                recorder.merge_files(SOUND_PATH_TMP, SOUND_PATH)
                return

    recorder.merge_files(SOUND_PATH_TMP, SOUND_PATH)


# --------------------------------
def posetmp():
    while True:
        print("\n\033[33m==================")
        print("|是否危險： " + str(is_danger_record) + "|")
        print("==================\033[37m\n")

        if not is_danger_record:
            get_skeleton()

        else:
            recording_danger()


# --------------------------------
# main
if __name__ == "__main__":
    detectesound()

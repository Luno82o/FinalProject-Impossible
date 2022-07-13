import cv2
import queue
import multiprocessing
import threading
import os


# --------------------------------------------------------
# read video type
class videoType(object):
    def __init__(self, path, interval=1):

        # 判斷路徑
        self.path = path
        self.interval = interval
        self.pathErr()

        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        ret, self.image = self.cap.read()

        self.isStop = False

    def pathErr(self):
        if not os.path.exists(self.path):
            print("-------------------------------")
            print("錯誤：找不到指定路徑影片檔")
            os._exit(0)

    def has_img(self):
        return self.image is not None

    def read_img(self):
        image = self.image
        for i in range(self.interval):
            ret, self.image = self.cap.read()

        return image

    def FPS(self):
        return self.fps

    def stop(self):
        self.cap.release()
        self.isStop = True

    def __del__(self):
        print("-------------------------------")
        print("[已關閉影片串流]")


# --------------------------------------------------------
# read image type
class imageType(object):
    def __init__(self, folder):
        self.now_img = 0
        self.len_img = 0

        # 判斷路徑
        self.folder = folder
        self.pathErr()

        self.img_path = [folder + '/' + path for path in os.listdir(folder)
                         if path.endswith('jpg') or path.endswith('png')]

    def pathErr(self):
        if not os.path.exists(self.folder):
            print("-------------------------------")
            print("錯誤：找不到指定路徑照片檔")
            os._exit(0)

        self.len_img = len(os.listdir(self.folder))
        if not self.len_img:
            print("-------------------------------")
            print("錯誤：該資料夾無照片檔")
            os._exit(0)

    def has_img(self):
        return self.now_img < self.len_img

    def read_img(self):
        img = cv2.imread(self.img_path[self.now_img])
        self.now_img += 1
        return img

    def FPS(self):
        return 10.0

    def stop(self):
        None

    def __del__(self):
        print("-------------------------------")
        print("[已關閉照片串流]")


# --------------------------------------------------------
# read webcam type
class webcamType(object):
    def __init__(self):

        self.cap = cv2.VideoCapture(0)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # multi-thread and multi-process
        self.img_queue = queue.Queue(maxsize=3)
        self.survive = multiprocessing.Value('i', True)
        self.thread = threading.Thread(target=self.thread_img)

        self.thread.start()

    def thread_img(self):
        while self.survive.value:
            ret, image = self.cap.read()
            if self.img_queue.full():  # if queue is full, pop one
                img_to_discard = self.img_queue.get()

            self.img_queue.put(image)

    def has_img(self):
        return True

    def read_img(self):
        img = self.img_queue.get()
        return img

    def FPS(self):

        return self.fps

    def stop(self):
        self.survive.value = False
        self.cap.release()

    def __del__(self):
        print("-------------------------------")
        print("[已關閉鏡頭與執行緒]")

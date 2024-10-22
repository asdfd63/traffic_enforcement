import sys
import cv2
import threading
import time

from ultralytics import YOLO
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap


def window_resize(self):
    global window_w, window_h
    window_w = Form.width()
    window_h = Form.height()
    label.setGeometry(video_x, video_y, window_w, window_h)


def close_opencv(self):
    """ 關閉視窗事件 """
    global ocv
    ocv = False  # 結束執行主程式


app = QtWidgets.QApplication(sys.argv)   # 視窗程式開始
window_w, window_h = 1920, 1080          # 定義預設長寬尺寸
video_x, video_y = 320, -200             # 定義預設影片座標

Form = QtWidgets.QWidget()               # 放入基底元件
Form.setWindowTitle('qt_main')           # 設定視窗標題
Form.resize(window_w, window_h)          # 設定視窗長寬尺寸

label = QtWidgets.QLabel(Form)                           # 放入 QLabel 用於顯示主程式執行結果
label.setGeometry(video_x, video_y, window_w, window_h)  # 設定 label 尺寸和位置
Form.resizeEvent = window_resize         # 縮放視窗時按照目前尺寸縮放
Form.closeEvent = close_opencv           # 關閉視窗事件發生時結束主程式

label2 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示 id
label2.setGeometry(400, 550, 1500, 400)  # 設定 label2 尺寸和位置

label3 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示 class
label3.setGeometry(400, 650, 1500, 400)  # 設定 label3 尺寸和位置

label4 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示影片路徑
label4.setGeometry(400, 750, 1500, 400)  # 設定 label4 尺寸和位置

font = QtGui.QFont()   # 加入文字設定
font.setPointSize(20)  # 設定文字大小
label2.setFont(font)   # label2 套用文字設定
label3.setFont(font)   # label3 套用文字設定
label4.setFont(font)   # label4 套用文字設定

ocv = True  # 是否執行主程式

filePath = []
filePath_run = []


# 辨識主程式
def opencv():
    global window_w, window_h, ocv, filePath
    model = YOLO('best.pt')
    video_path = filePath[0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    while ocv:
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame")
            break

        results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml")
        frame_ = results[0].plot()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            output = str(track_ids).strip('[]')
            label2.setText('id: ' + output)

        if results[0].boxes.cls is not None:
            track_cls = results[0].boxes.cls.int().cpu().tolist()
            track_cls2 = []
            for i in range(len(track_cls)):
                if track_cls[i] == 0:
                    track_cls2.append('bus')
                elif track_cls[i] == 1:
                    track_cls2.append('car')
                elif track_cls[i] == 2:
                    track_cls2.append('moto')
            output2 = str(track_cls2).strip('[]')
            label3.setText('class: ' + output2)

        frame = cv2.resize(frame_, (1280, 720))   # 改變影像尺寸
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 影像轉換成 RGB
        height, width, channel = frame.shape            # 讀取影像尺寸和 channel 數量
        bytesPerline = channel * width                  # 設定 bytesPerline (轉換使用)
        img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)  # 轉換影像為 QImage，讓 PyQt5 可以讀取
        label.setPixmap(QPixmap.fromImage(img))         # QLabel 顯示影像


# 開啟影片
def open_video():
    global filePath, filePath_run
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileNames()  # 選擇檔案對話視窗
    # 顯示影片路徑
    if filePath_run != filePath:
        label4.setText('path: ' + filePath[0] + ' is not running')
    else:
        label4.setText('path: ' + filePath[0] + ' is running')


# 執行辨識
def run_yolo():
    global ocv, filePath_run
    ocv = False  # 結束執行緒
    time.sleep(0.1)
    ocv = True
    video = threading.Thread(target=opencv)  # 建立執行緒
    video.start()                            # 啟用執行緒
    # 顯示影片路徑
    filePath_run = filePath[0]
    if filePath_run != filePath[0]:
        label4.setText('path: ' + filePath[0] + ' is not running')
    else:
        label4.setText('path: ' + filePath[0] + ' is running')


# 放入按鈕 1 並設定參數
btn1 = QtWidgets.QPushButton(Form)
btn1.setText('open')
btn1.setFont(font)
btn1.setGeometry(50, 60, 100, 60)
btn1.clicked.connect(open_video)  # 點擊按鈕執行 open_video 函式

# 放入按鈕 2 並設定參數
btn2 = QtWidgets.QPushButton(Form)
btn2.setText('run')
btn2.setFont(font)
btn2.setGeometry(160, 60, 100, 60)
btn2.clicked.connect(run_yolo)    # 點擊按鈕執行 run_yolo 函式

Form.show()             # 顯示元件
sys.exit(app.exec_())   # 視窗程式結束

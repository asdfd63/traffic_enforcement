import sys
import time as time_module
import psutil
import cv2
import numpy as np
import supervision as sv
import queue
import threading
import copy
import logging
import pymongo
import math

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
from typing import List
from datetime import datetime
from detect import estimate_label

# 日誌配置
LOG_FILE = "fail.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode='w',       # 寫入模式，清空舊內容
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logging.info("Program started.")

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])             # 建立調色盤
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)                                           # 建立顏色註解器
LABEL_ANNOTATOR = sv.LabelAnnotator(color=COLORS, text_color=sv.Color.from_hex("#000000"))  # 建立標籤註解器

ocv = False        # 是否進行辨識
cur_frame = -1     # 經過影格數，進迴圈後加一使其從零開始
interval = 7       # 採樣間隔

# 連結 mongoDB 資料庫
myclient = pymongo.MongoClient("mongodb://localhost:27017/")
mydb = myclient["mydatabase"]
pos_hist = mydb["pos_hist"]
pos_hist.delete_many({})

# 建立佇列存放影格
q = queue.Queue()   # 用於顯示畫面
q1 = queue.Queue()  # 用於越線截圖
q2 = queue.Queue()  # 用於違停錄影
q3 = queue.Queue()  # 用於違停錄影

# 違規停車
start_pos = {}      # 記錄車輛首次進入區域之影格位置
end_pos = {}        # 記錄車輛停留達到指定時間之影格位置
recorded = {}       # 記錄此車輛是否完成違停錄影

# 違規迴轉
wup = {}            # 記錄進入第一個區域車輛
wrongway = []       # 記錄違規迴轉車輛
start_pos2 = {}     # 記錄車輛首次進入區域之影格位置
end_pos2 = {}       # 記錄車輛停留達到指定時間之影格位置
recorded2 = {}      # 記錄此車輛是否完成迴轉錄影

car_crossed = {}    # 記錄越界車輛

# 記錄車輛位於停止線哪一側
pre_side = {}
cur_side = {}

# 儲存區域及線段
vertex = []
types = []
START = None
END = None
area1 = None
area2 = None


def xyxy_to_xywh(box) -> list:
    """轉換 [x1 y1 x2 y2] 為 [x y w h] 格式"""
    x_min, y_min, x_max, y_max = box  # 左上角 (x_min, y_min) 右下角 (x_max, y_max)
    x_center = (x_min + x_max) / 2  # 中心點 x 座標
    y_center = (y_min + y_max) / 2  # 中心點 y 座標
    w = x_max - x_min  # 寬度
    h = y_max - y_min  # 高度
    return [int(x_center), int(y_center), w, h]


def img_crop(frame, xx1, yy1, ww, hh, zoom) -> np.ndarray:
    """以指定倍率截取圖片 (xywh 格式)"""
    x1 = int(xx1 - ww * (zoom - 1) / 2)
    y1 = int(yy1 - hh * (zoom - 1) / 2)
    w = int(ww * zoom)
    h = int(hh * zoom)
    return frame[y1:y1 + h, x1:x1 + w]


def find_point_side(x1, y1, x2, y2, cx, cy) -> bool:
    """判斷點位於線段的哪一側"""
    # 計算斜率 m
    if x2 - x1 == 0:
        raise ValueError("兩點的x值相同，斜率不存在（垂直線）")
    m = (y2 - y1) / (x2 - x1)

    # 計算截距 b
    b = y1 - m * x1

    # 計算點為正數或負數
    d = m * cx - cy + b

    return d > 0


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.window_w = 1920
        self.window_h = 1080
        self.video_x = 320
        self.video_y = -100

        self.zone_configuration_path = "config.json"
        self.type_configuration_path = "config2.json"
        self.rtsp_url = "rtsp://localhost:554/s"
        self.weights = "best.pt"
        self.device = "cuda"
        self.confidence = 0.3
        self.iou = 0.7
        self.classes = []

        self.receive_thread = None
        self.display_thread = None

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('qt_main')  # 設定視窗標題
        self.resize(self.window_w, self.window_h)
        self.create_menu()
        self.create_label()
        self.create_button()

    def create_menu(self):
        # 創建菜單欄
        self.menubar = QtWidgets.QMenuBar(self)

        # 創建 File 菜單
        self.menu_file = QtWidgets.QMenu('File')

        # 創建 start 動作
        self.action_start = QtWidgets.QAction('Start')
        self.action_start.triggered.connect(self.start)
        self.menu_file.addAction(self.action_start)

        # 創建 stop 動作
        self.action_stop = QtWidgets.QAction('Stop')
        self.action_stop.triggered.connect(self.stop)
        self.menu_file.addAction(self.action_stop)

        # 創建 Close 動作
        self.action_close = QtWidgets.QAction('Close')
        self.action_close.triggered.connect(QtWidgets.QApplication.instance().quit)
        self.menu_file.addAction(self.action_close)

        # 將 File 菜單添加到菜單欄
        self.menubar.addMenu(self.menu_file)

    def create_label(self):
        self.video = QtWidgets.QLabel(self)  # 放入 QLabel 用於顯示執行結果
        self.video.setGeometry(self.video_x, self.video_y, self.window_w, self.window_h)

        self.label2 = QtWidgets.QLabel(self)  # 放入 QLabel 顯示 id
        self.label2.setGeometry(400, 650, 1500, 400)

        self.label3 = QtWidgets.QLabel(self)  # 放入 QLabel 顯示 class
        self.label3.setGeometry(400, 700, 1500, 400)

        self.label4 = QtWidgets.QLabel(self)  # 放入 QLabel 顯示
        self.label4.setGeometry(400, 750, 1500, 400)

        font = QtGui.QFont()  # 加入文字設定
        font.setPointSize(20)  # 設定文字大小
        # label 套用文字設定
        self.label2.setFont(font)
        self.label3.setFont(font)
        self.label4.setFont(font)

    def create_button(self):
        # 放入按鈕 1 並設定參數
        self.btn1 = QtWidgets.QPushButton(self)
        self.btn1.setText('start')
        self.btn1.setGeometry(50, 60, 100, 60)
        self.btn1.clicked.connect(self.start)  # 點擊按鈕執行 start 函式

        # 放入按鈕 2 並設定參數
        self.btn2 = QtWidgets.QPushButton(self)
        self.btn2.setText('stop')
        self.btn2.setGeometry(160, 60, 100, 60)
        self.btn2.clicked.connect(self.stop)  # 點擊按鈕執行 end 函式

        font = QtGui.QFont()  # 加入文字設定
        font.setPointSize(20)  # 設定文字大小
        self.btn1.setFont(font)
        self.btn2.setFont(font)

    def resizeEvent(self, event):
        """自適應視窗尺寸"""
        super().resizeEvent(event)
        self.window_w, self.window_h = event.size().width(), event.size().height()
        self.video.setGeometry(self.video_x, self.video_y, self.window_w, self.window_h)

    def closeEvent(self, event):
        """關閉視窗結束執行主程式"""
        self.stop()
        logging.info("Window closed.")

    def receive(self, rtsp_url: str):
        """接收 RTSP 影像串流"""
        logging.info("Receive thread started.")
        try:
            cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
            if not cap.isOpened():
                logging.error("Failed to open RTSP stream.")
                return

            logging.info("RTSP stream opened successfully.")

            ret, frame = cap.read()  # 讀取首個影格
            q.put(frame)             # 將影格放入佇列

            # 計算 fps (一秒經過影格數)
            last_frame = 0
            last_time = int(datetime.now().timestamp() * 1000)
            while ocv:
                cur_time = int(datetime.now().timestamp() * 1000)
                if cur_time - last_time > 1000:  # 當相差 1000 ms
                    real_fps = cur_frame - last_frame
                    self.label4.setText(f"fps: {real_fps}")
                    last_frame = cur_frame
                    last_time = cur_time

                ret = cap.grab()  # 從視訊檔案或攝影機抓取下一影格，並在成功的情況下回傳 True
                if not ret:       # 當串流停止
                    logging.warning("Failed to retrieve frame. Stopping receive thread.")
                    break
                if cap.get(1) % interval == 0:   # 跳幀處理
                    ret, image = cap.retrieve()  # 解碼並回傳下一個影格
                    q.put(image)                 # 將影格放入佇列

            cap.release()  # 串流結束釋放資源
            logging.info("Receive thread stopped.")
        except Exception as e:
            logging.error(f"Error in receive: {e}")

    def display(self, device: str, confidence: float, iou: float, classes: List[int],
                model: YOLO, tracker: any, zones: list, timers: list, fps: float, base_time: int):
        """處理並顯示影像"""
        global ocv, cur_frame
        parking_time = 10  # 定義違停秒數
        max_qsize = 310     # 佇列最大儲存量 (parking_time * 30 以上)
        offset = 0         # 當前影格索引向前位移量
        crossed_cnt = 0    # 越線車輛數

        if model is None:
            # 錯誤：YOLO 模型未初始化
            logging.error("YOLO model is not initialized. Exiting display thread.")
            return

        logging.info("Display thread started.")
        try:
            while ocv:
                if not q.empty():  # 若佇列不為空
                    # 檢查可用記憶體
                    memory_info = psutil.virtual_memory()
                    available_memory = memory_info.available / (1024 ** 3)
                    if available_memory < 0.1:
                        # 警告：可用記憶體低於 0.1 GB，立即中斷運行
                        self.label2.setText(f"The available memory is less than 0.1 GB")
                        logging.warning(
                            "The available memory is less than 0.1 GB, and the execution is automatically terminated.")
                        self.stop()

                    # 超過限制大小後，移除佇列中最舊影格，其餘往前移動
                    if q1.qsize() >= max_qsize:
                        q1.get()
                        q2.get()
                        q3.get()
                        offset += 1

                    frame = q.get()  # 取出影格

                    try:
                        results = model(frame, verbose=False, device=device, conf=confidence)[0]      # 使用 YOLOv8 推理
                        detections = sv.Detections.from_ultralytics(results)             # 根據 YOLOv8 推理結果建立檢測實例
                        detections = detections[find_in_list(detections.class_id, classes)]    # 選擇僅屬於選定類別集的偵測
                        detections = detections.with_nms(threshold=iou)                          # 對檢測集執行非極大值抑制
                        detections = tracker.update_with_detections(detections)  # 使用提供的偵測更新追蹤器並回傳更新的偵測結果
                    except Exception as e:
                        logging.error(f"Error during YOLO inference: {e}")
                        continue

                    # 建立影格副本用於標註
                    annotated_frame = frame.copy()
                    annotated_frame2 = frame.copy()

                    cur_frame += 1                                   # 經過影格數加一
                    duration = round(cur_frame / fps * interval, 1)  # 經過秒數
                    cur_sec = base_time + duration                   # 實際時間 (秒數)
                    cur_time = datetime.fromtimestamp(cur_sec)       # 實際時間 (完整)
                    cur_time = cur_time.strftime("%Y-%m-%d %H:%M:%S") + f".{int(cur_time.microsecond / 100000)}"  # 格式化時間
                    cv2.putText(annotated_frame, f"{cur_time}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 3)
                    cv2.putText(annotated_frame2, f"{cur_time}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cur_time2 = datetime.fromtimestamp(cur_sec).strftime("%Y-%m-%d,%H-%M-%S")  # 格式化時間
                    # 取得此影格之物件邊界框, ID, 類別 (ndarray)
                    boxes = detections.xyxy
                    track_ids = detections.tracker_id
                    track_cls = detections.class_id

                    # 燈號檢測
                    light_type = "unknown"
                    for idx, cls in enumerate(types):
                        if cls == 1:
                            # 左上座標
                            x1 = vertex[idx][0][0].item()
                            y1 = vertex[idx][0][1].item()
                            # 右下座標
                            x2 = vertex[idx][2][0].item()
                            y2 = vertex[idx][2][1].item()
                            w = x2 - x1  # 寬度
                            h = y2 - y1  # 高度
                            light_img = img_crop(frame, x1, y1, w, h, 1)                     # 擷取紅綠燈部分
                            light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)                 # 轉換色彩空間
                            light_type = estimate_label(light_img, cur_frame, False)        # 估計燈號
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)   # 標示紅綠燈
                            cv2.rectangle(annotated_frame2, (x1, y1), (x2, y2), (255, 0, 255), 2)  # 標示紅綠燈

                    # 當物件列表不為空，逐一遍歷影格中物件
                    if len(track_ids) > 0:
                        for box, ID, cls in zip(boxes, track_ids, track_cls):
                            ID = ID.item()
                            x1, y1, x2, y2 = box.astype(int)
                            cur_pos = [x1.item(), y1.item(), x2.item(), y2.item()]  # 物件在此影格之座標
                            cx, cy, w, h = xyxy_to_xywh(box)
                            cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)

                            id_exist = pos_hist.find_one({'_id': ID})
                            if not id_exist:
                                cur_id = {'_id': ID, 'info': [[cur_frame, cur_pos]]}
                                pos_hist.insert_one(cur_id)
                            else:
                                mydoc = pos_hist.find({"_id": ID})
                                for x in mydoc:
                                    temp = x['info']
                                    old = {'info': temp.copy()}
                                    temp.append([cur_frame, cur_pos])
                                    new = {"$set": {'info': temp}}
                                    pos_hist.update_one(old, new)

                            # 檢查違規迴轉
                            # 偵測物件是否在 area1 中
                            if area1 is not None:
                                result = cv2.pointPolygonTest(area1, (cx, cy), False)
                                if result >= 0:
                                    wup[ID] = (cx, cy)
                                    if ID not in start_pos2:
                                        start_pos2[ID] = cur_frame

                            # 如果物件已經在 area1，檢查是否進入 area2
                            if ID in wup and area2 is not None:
                                result1 = cv2.pointPolygonTest(area2, (cx, cy), False)
                                if result1 >= 0:
                                    # 標註違規車輛
                                    cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 0), -1)
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                    cv2.putText(annotated_frame, f'ID:{ID}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                    if ID not in wrongway:
                                        wrongway.append(ID)
                                        end_pos2[ID] = cur_frame

                            # 檢查物件是否越線 (不分車種)
                            if START and END and ((START.x < cx < END.x or START.x > cx > END.x) or (START.y < cy < END.y or START.y > cy > END.y)):
                                cur_side[ID] = find_point_side(START.x, START.y, END.x, END.y, cx, cy)
                                if ID in pre_side and cur_side[ID] != pre_side[ID]:
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                    if ID not in car_crossed:
                                        car_crossed[ID] = [True, cur_frame, 0]
                                        crossed_cnt += 1
                                pre_side[ID] = cur_side[ID]
                            # 離開線段範圍後移除
                            elif ID in pre_side:
                                pre_side.pop(ID)

                    # 繪製違規迴轉區域
                    if area1 is not None:
                        cv2.polylines(annotated_frame, [area1], True, (255, 255, 255), 2)
                    if area2 is not None:
                        cv2.polylines(annotated_frame, [area2], True, (255, 255, 255), 2)

                    q3.put(copy.deepcopy(annotated_frame))

                    # 繪製停止線
                    if START and END:
                        cv2.line(img=annotated_frame, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)
                        cv2.line(img=annotated_frame2, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)

                    q1.put(copy.deepcopy(annotated_frame2))

                    # 繪製違停區域及標註違規車輛
                    for idx, zone in enumerate(zones):  # 將可迭代的對象（如列表、元組或字串）轉換為索引序列，同時列出資料和資料對應的索引值 (idx)
                        annotated_frame = sv.draw_polygon(scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx))  # 在畫面上繪製多邊形

                        if len(track_ids) > 0:
                            detections_in_zone = detections[zone.trigger(detections)]              # 將檢測與 PolygonZone 結合使用來清除區域內外的邊界框
                            time_in_zone = timers[idx].tick(detections_in_zone)                    # 處理目前影格，更新每個追蹤器的持續時間
                            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)  # 傳回給定形狀和類型的新陣列，並用 idx 填滿，用於定義將顏色對應到註解的策略的枚舉類別

                            # 用顏色註解場景中區域
                            annotated_frame = COLOR_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                custom_color_lookup=custom_color_lookup,
                            )

                            # 建立標籤(ID, 時間)
                            labels = []
                            for box, ID, time in zip(detections_in_zone.xyxy, detections_in_zone.tracker_id, time_in_zone):
                                cx, cy, w, h = xyxy_to_xywh(box)
                                real_time = time * interval  # 由於跳格處理
                                labels.append(f"#{ID} {int(real_time // 60):02d}:{int((real_time % 60)):02d}")
                                # 記錄車輛首次進入區域之影格位置
                                if ID not in start_pos:
                                    start_pos[ID] = cur_frame
                                # 記錄車輛停留達到指定時間之影格位置
                                if real_time > parking_time and ID not in end_pos:
                                    end_pos[ID] = [cur_frame, [cx, cy]]

                            # 用標籤註解畫面中區域
                            annotated_frame = LABEL_ANNOTATOR.annotate(
                                scene=annotated_frame,
                                detections=detections_in_zone,
                                labels=labels,
                                custom_color_lookup=custom_color_lookup,
                            )

                    q2.put(copy.deepcopy(annotated_frame))

                    # 顯示資訊於畫面
                    cv2.putText(annotated_frame, f"Sec: {duration} s", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 3)
                    cv2.putText(annotated_frame, f"Light: {light_type}", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cv2.putText(annotated_frame, f"Car crossed: {crossed_cnt}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 3)
                    cv2.putText(annotated_frame, f"Wrong-way Cars: {len(wrongway)}", (10, 370), cv2.FONT_HERSHEY_SIMPLEX, 2,(255, 255, 255), 3)

                    # 越線截圖
                    if car_crossed:
                        for ID in list(car_crossed):
                            q1_list = list(q1.queue)  # queue 轉換為 list
                            img_list = []             # 存放不同位置之圖片
                            car_crossed[ID][2] += 1   # 通過停止線後經過影格數 +1
                            delay = 10                # 通過停止線後第 n 個影格開始往前記錄
                            id_exist = pos_hist.find_one({'_id': ID})
                            if car_crossed[ID][2] == delay and id_exist:
                                index_map = {}
                                for i in [int(delay / -1.5 * 3), int(delay / -1.5 * 2), int(delay / -1.5), 0]:
                                    # 在前面四個不同位置擷取影格
                                    pos = cur_frame + i - offset  # 影格索引值
                                    if 0 <= pos <= len(q1_list):  # 邊界檢查
                                        x = pos_hist.find_one({"_id": ID})
                                        for idx, item in enumerate(x['info']):
                                            if len(item) > 1:
                                                index_map[item[0]] = idx
                                        index = index_map.get(pos + offset)
                                        if index is not None:
                                            x1, y1, x2, y2 = x['info'][index][1]
                                            cv2.rectangle(q1_list[pos], (x1, y1), (x2, y2), (0, 255, 0), 2)
                                            cv2.putText(q1_list[pos], f'ID:{ID}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                        cv2.putText(q1_list[pos], f"{len(img_list) + 1}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                                        img_list.append(q1_list[pos])
                                        if len(img_list) == 2:
                                            img_hor1 = cv2.hconcat([img_list[0], img_list[1]])  # 上方左右水平拼接
                                        elif len(img_list) == 4:
                                            img_hor2 = cv2.hconcat([img_list[2], img_list[3]])  # 下方左右水平拼接
                                            img_ver = cv2.vconcat([img_hor1, img_hor2])         # 上下方垂直拼接
                                            filename = f"save/line_{str(cur_time2)}_{ID}.jpg"
                                            cv2.imwrite(filename, img_ver)  # 儲存圖片
                                car_crossed.pop(ID)  # 從紀錄移除

                    # 違停錄影
                    if end_pos:
                        for ID, pos in end_pos.items():
                            is_same = False
                            if ID not in recorded:
                                for ID2, ctr in recorded.items():
                                    x1 = pos[1][0]
                                    y1 = pos[1][1]
                                    x2 = ctr[0]
                                    y2 = ctr[1]
                                    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    if distance < 150:
                                        is_same = True
                                if not is_same:
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片的格式為 mp4
                                    out = cv2.VideoWriter(f"save/park_{str(cur_time2)}_{ID}.mp4", fourcc, fps / interval, (1920, 1080))  # 產生空影片
                                    q2_list = list(q2.queue)        # queue 轉換為 list
                                    start = start_pos[ID] - offset  # 進入區域之影格位置
                                    end = pos[0] - offset           # 確定違停之影格位置
                                    for i in range(start, end):     # 遍歷中間經過之影格
                                        out.write(q2_list[i])       # 將影格寫入影片
                                    recorded[ID] = pos[1]           # 此車輛已完成錄影
                                    out.release()                   # 釋放資源

                    # 迴轉錄影
                    if end_pos2:
                        for ID, pos in end_pos2.items():
                            if ID not in recorded2:
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片的格式為 mp4
                                out = cv2.VideoWriter(f"save/turn_{str(cur_time2)}_{ID}.mp4", fourcc, fps / interval, (1920, 1080))  # 產生空影片
                                q3_list = list(q3.queue)         # queue 轉換為 list
                                start = start_pos2[ID] - offset  # 進入 area1 之影格位置
                                end = pos - offset               # 進入 area2 之影格位置
                                for i in range(start, end):      # 遍歷中間經過之影格
                                    out.write(q3_list[i])        # 將影格寫入影片
                                recorded2[ID] = True             # 此車輛已完成錄影
                                out.release()                    # 釋放資源

                    # 顯示資訊於視窗
                    self.label2.setText(f"Available RAM: {available_memory:.2f} GB")
                    self.label3.setText(f"qsize: {q.qsize()},  time: {duration} s")

                    frame = cv2.resize(annotated_frame, (1280, 720))  # 調整影像尺寸
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # 影像轉換成 RGB
                    height, width, channel = frame.shape                    # 讀取影像尺寸和 channel 數量
                    bytes_per_line = channel * width                        # 設定 bytes_Per_line 用於轉換
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)  # 轉換影像為 QImage，使 PyQt5 可以讀取
                    self.video.setPixmap(QPixmap.fromImage(q_image))        # label 顯示影像
                else:
                    time_module.sleep(0.01)  # 等待 0.01 秒
            logging.info("Display thread stopped.")
        except Exception as e:
            logging.error(f"Error in display: {e}")

    def start(self):
        """初始化參數後執行程式"""
        global ocv, vertex, types, START, END, area1, area2
        logging.info("Program start button clicked.")

        if ocv:
            # 警告：程式已經在運作。忽略重複的啟動命令
            logging.warning("Program is already running. Ignoring duplicate start command.")
            return

        try:
            vertex = load_zones_config(file_path=self.zone_configuration_path)  # 從 JSON 檔案載入多邊形區域配置
            types = load_zones_config(file_path=self.type_configuration_path)   # 代號：1 紅綠燈, 2 停止線, 3 迴轉區, 4 臨停區

            if not vertex or not types:
                # 錯誤：無法載入 JSON 配置
                self.label2.setText("Error: Failed to load JSON configuration.")
                logging.error("Failed to load JSON configuration. Cannot proceed.")
                return

            try:
                model = YOLO(self.weights)  # 初始化 YOLO 模型
                logging.info(f"Loaded YOLO model from {self.weights}")
            except Exception as e:
                logging.error(f"Failed to load YOLO model: {e}")
                self.label2.setText("Error: Failed to load YOLO model.")
                return

            ocv = True  # 開始辨識

            cap = cv2.VideoCapture(self.rtsp_url)  # 擷取串流影像
            if not cap.isOpened():
                # 錯誤：開啟串流失敗
                logging.error("Failed to open RTSP stream.")
                return
            fps = cap.get(cv2.CAP_PROP_FPS)  # 取得串流之 fps
            tracker = sv.ByteTrack(frame_rate=round(fps), track_activation_threshold=self.confidence)  # 初始化 ByteTrack 物件
            cap.release()  # 釋放資源

            # 定義違停區
            polygons = []
            for idx, cls in enumerate(types):
                if cls == 4:
                    polygons.append(vertex[idx])

            # 定義違停區域以偵測停留之物件
            zones = [
                sv.PolygonZone(
                    polygon=polygon,                           # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 xy 座標
                    triggering_anchors=(sv.Position.CENTER,),  # 位置列表，指定在決定檢測是否通過線計數器時要考慮的檢測邊界框的錨點。預設情況下，這包含檢測邊界框的四個角
                )
                for polygon in polygons
            ]

            timers = [FPSBasedTimer(round(fps)) for _ in zones]  # 對每個區域使用指定的 fps 初始化計時器物件

            # 定義停止線
            for idx, cls in enumerate(types):
                if cls == 2:
                    START = sv.Point(vertex[idx][0][0].item(), vertex[idx][0][1].item())
                    END = sv.Point(vertex[idx][1][0].item(), vertex[idx][1][1].item())

            # 定義區域 area1 和 area2，以檢測違規迴轉行為
            area_cnt = 1
            for idx, cls in enumerate(types):
                if cls == 3:
                    # 因迴轉區域成對，假設奇數個為 area1，偶數個為 area2
                    if area_cnt % 2:
                        area1 = np.array(vertex[idx], np.int32)
                    else:
                        area2 = np.array(vertex[idx], np.int32)
                    area_cnt += 1

            base_sec = int(datetime.now().timestamp())  # 記錄開始辨識之時間

            # 多執行緒的目的是讓串流接收不阻塞主執行緒
            # 守護執行緒 (Daemon Thread)：當主程式結束時，守護執行緒會自動結束，而非守護執行緒則會繼續執行，直到完成自己的任務

            # 啟動接收執行緒
            self.receive_thread = threading.Thread(target=self.receive, args=(self.rtsp_url,))
            self.receive_thread.daemon = True
            self.receive_thread.start()

            # 啟動顯示執行緒
            self.display_thread = threading.Thread(target=self.display, args=(self.device, self.confidence, self.iou, self.classes, model, tracker, zones, timers, fps, base_sec,))
            self.display_thread.daemon = True
            self.display_thread.start()

            logging.info("Threads started.")
        except Exception as e:
            logging.error(f"Error in start: {e}")

    def stop(self):
        """停止程式"""
        global ocv
        if not ocv:
            # 警告：程式沒有運行。忽略停止命令。
            logging.warning("Program is not running. Ignoring stop command.")
            return

        ocv = False  # 停止辨識
        logging.info("Program stopped by user.")

        # 檢查並處理執行緒的狀態，確保所有正在運行的執行緒能夠在應用程式關閉之前被安全地處理
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1)
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1)


if __name__ == "__main__":
    # 若程式是直接執行，而非被其他模組匯入，則啟動以下程式
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        logging.info("Qt application started.")
        sys.exit(app.exec_())
    except Exception as e:
        logging.error(f"Unhandled exception: {e}")

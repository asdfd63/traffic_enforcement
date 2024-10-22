import sys
import time

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap

import argparse
from typing import List
import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer
import supervision as sv
from detect import estimate_label
import queue
import threading


def window_resize(self):
    """ 自適應視窗尺寸 """
    global window_w, window_h
    window_w = Form.width()
    window_h = Form.height()
    label.setGeometry(video_x, video_y, window_w, window_h)


def close_opencv(self):
    """ 關閉視窗結束執行主程式 """
    global ocv
    ocv = False


app = QtWidgets.QApplication(sys.argv)   # 視窗程式開始
window_w, window_h = 1920, 1080          # 定義預設長寬尺寸
video_x, video_y = 320, -100             # 定義預設影片座標

Form = QtWidgets.QWidget()               # 放入基底元件
Form.setWindowTitle('qt_main')           # 設定視窗標題
Form.resize(window_w, window_h)          # 設定視窗長寬尺寸

label = QtWidgets.QLabel(Form)                           # 放入 QLabel 用於顯示主程式執行結果
label.setGeometry(video_x, video_y, window_w, window_h)  # 設定 label 尺寸和位置
Form.resizeEvent = window_resize         # 縮放視窗時按照目前尺寸縮放
Form.closeEvent = close_opencv           # 關閉視窗事件發生時結束主程式

label2 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示 id
label2.setGeometry(400, 650, 1500, 400)  # 設定 label2 尺寸和位置

label3 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示 class
label3.setGeometry(400, 700, 1500, 400)  # 設定 label3 尺寸和位置

label4 = QtWidgets.QLabel(Form)          # 放入 QLabel 顯示
label4.setGeometry(400, 750, 1500, 400)  # 設定 label4 尺寸和位置

font = QtGui.QFont()   # 加入文字設定
font.setPointSize(20)  # 設定文字大小
label2.setFont(font)   # label2 套用文字設定
label3.setFont(font)   # label3 套用文字設定
label4.setFont(font)   # label4 套用文字設定

q = queue.Queue()  # 建立佇列存放影格
ocv = True         # 是否執行主程式

cur_time = 0   # 經過時間
frame_pos = 0  # 經過影格數
crop_cnt = 0   # 已截圖數
crop_max = 0   # 截圖最大數

# 建立字典來追蹤越界的對象
car_crossed = {}
moto_crossed = {}

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])  # 建立調色盤
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)  # 建立顏色註解器
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)  # 建立標籤註解器


def xyxy_to_xywh(box) -> list:
    """ 轉換 [x1 y1 x2 y2] 為 [x y w h] 格式。 """
    x_min, y_min, x_max, y_max = box  # 左上角 (x_min, y_min) 右下角 (x_max, y_max)
    x_center = (x_min + x_max) / 2  # 中心點 x 座標
    y_center = (y_min + y_max) / 2  # 中心點 y 座標
    w = x_max - x_min  # 寬度
    h = y_max - y_min  # 高度
    return [x_center, y_center, w, h]


def img_crop(frame, xx1, yy1, ww, hh, zoom) -> np.ndarray:
    """ 以倍率截圖 (xywh 格式) """
    x1 = int(xx1 - ww * (zoom - 1) / 2)
    y1 = int(yy1 - hh * (zoom - 1) / 2)
    w = int(ww * zoom)
    h = int(hh * zoom)
    return frame[y1:y1 + h, x1:x1 + w]


def receive(rtsp_url: str, fps: float) -> None:
    """ 讀取影格 """
    global frame_pos, cur_time, ocv
    print("\nStart receive")

    cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
    ret, frame = cap.read()           # 讀取首個影格
    q.put(frame)                      # 將影格放入佇列
    while ret and ocv:                        # 當串流進行中
        cur_time = round(frame_pos / fps, 1)  # 計算經過秒數
        frame_pos += 1                        # 計算經過影格數
        ret = cap.grab()                      # 從視訊檔案或攝影機抓取下一影格，並在成功的情況下回傳 True
        if frame_pos % 2 == 0:           # 跳幀處理
            ret, image = cap.retrieve()  # 解碼並回傳下一個影格
            q.put(image)                 # 將影格放入佇列

    # 串流結束釋放資源
    cap.release()
    print("End receive")


def display(device: str, confidence: float, iou: float, classes: List[int],
            model: YOLO, tracker: any, zones: list, timers: list, START: any, END: any) -> None:
    """ 接收影格處理後顯示 """
    global cur_time, frame_pos, crop_cnt, crop_max, ocv
    print("Start display")

    while ocv:  # 開始主程式
        if not q.empty():                                                             # 若佇列不為空
            frame = q.get()                                                           # 從佇列取出影格
            results = model(frame, verbose=False, device=device, conf=confidence)[0]  # 使用 YOLOv8 推理
            detections = sv.Detections.from_ultralytics(results)                      # 根據 YOLOv8 推理結果建立檢測實例
            detections = detections[find_in_list(detections.class_id, classes)]       # 選擇僅屬於選定類別集的偵測
            detections = detections.with_nms(threshold=iou)                           # 對檢測集執行非極大值抑制
            detections = tracker.update_with_detections(detections)                   # 使用提供的偵測更新追蹤器並回傳更新的偵測結果

            # 建立此影格的副本
            annotated_frame = frame.copy()

            # 取得物件邊界框和軌跡 ID
            boxes = detections.xyxy
            track_ids = detections.tracker_id
            track_cls = detections.class_id

            # 燈號檢測
            light_img = img_crop(frame, 1345, 311, 70, 19, 1)
            light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)
            light_type = estimate_label(light_img, frame_pos, False)

            # 繪製軌跡並計算越線物件的數量
            for box, id, cls in zip(boxes, track_ids, track_cls):
                x, y, w, h = xyxy_to_xywh(box)
                x1, y1, x2, y2 = box

                # 檢查物件是否越線
                if START.x < x < END.x and abs(y - START.y) < 5:  # 當物件水平交叉
                    # 當物件越過線時對其進行註釋
                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 1)

                    if id not in car_crossed:
                        if cls == 1:  # 0:bus 1:car 2:moto
                            car_crossed[id] = True
                            if crop_cnt < crop_max:  # 限定截圖次數
                                crop_img = img_crop(annotated_frame, x1, y1, h, w, zoom=1.5)
                                filename = "save/" + str(cur_time) + 's_ID(' + str(id) + ').jpg'
                                cv2.imwrite(filename, crop_img)
                                crop_cnt += 1

                    if id not in moto_crossed:
                        if cls == 2:  # 0:bus 1:car 2:moto
                            moto_crossed[id] = True
                            if crop_cnt < crop_max:  # 限定截圖次數
                                crop_img = img_crop(annotated_frame, x1, y1, w, h, zoom=3)
                                filename = "save/" + str(cur_time) + 's_ID(' + str(id) + ').jpg'
                                cv2.imwrite(filename, crop_img)
                                crop_cnt += 1

            # 繪製線段
            cv2.line(img=annotated_frame, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)

            # 繪製違停區域及標註違規車輛
            for idx, zone in enumerate(zones):  # 將可迭代的對象（如列表、元組或字串）轉換為索引序列，同時列出資料和資料對應的索引值 (idx)
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                )  # 在場景上繪製多邊形

                detections_in_zone = detections[zone.trigger(detections)]  # 將檢測與 PolygonZone 結合使用來清除區域內外的邊界框
                time_in_zone = timers[idx].tick(detections_in_zone)        # 處理目前影格，更新每個追蹤器的持續時間
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)
                # 傳回給定形狀和類型的新陣列，並用 idx 填滿，用於定義將顏色對應到註解的策略的枚舉類別

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )  # 用顏色註解場景中區域
                labels = [
                    f"#{id} {int(time * 2 // 60):02d}:{int((time * 2 % 60)):02d}"
                    for id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]  # 建立標籤 (將id.時間結合) (跳格處理因此時間需x2)
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )  # 用標籤註解場景中區域

            # 顯示統計資訊於畫面
            cv2.putText(annotated_frame, f"Car crossed: {len(car_crossed)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Moto crossed: {len(moto_crossed)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Time: {cur_time}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2)
            cv2.putText(annotated_frame, f"Light: {light_type}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)

            # 顯示統計資訊於視窗
            label2.setText(f"Total crossed: {len(car_crossed) + len(moto_crossed)}")
            label3.setText(f"Time: {cur_time}s")
            label4.setText(f"Light: {light_type}")

            frame = cv2.resize(annotated_frame, (1280, 720))  # 改變影像尺寸
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # 影像轉換成 RGB
            height, width, channel = frame.shape                    # 讀取影像尺寸和 channel 數量
            bytesPerline = channel * width                          # 設定 bytesPerline (用於轉換)
            img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)  # 轉換影像為 QImage，讓 PyQt5 可以讀取
            label.setPixmap(QPixmap.fromImage(img))                 # label 顯示影像

    print("End display")


def start():
    """ 執行主程式 """
    global ocv, is_display, is_receive

    model = YOLO(weights)             # 初始化 YOLO 模型
    cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
    fps = cap.get(cv2.CAP_PROP_FPS)   # 取得串流之 fps
    tracker = sv.ByteTrack(frame_rate=round(fps), track_activation_threshold=confidence)  # 初始化 ByteTrack 物件

    # 定義線段
    points = load_zones_config(file_path=zone_configuration_path)
    START = sv.Point(points[0][0][0].item(), points[0][0][1].item())
    END = sv.Point(points[0][1][0].item(), points[0][1][1].item())

    polygons = load_zones_config(file_path=zone_configuration_path, )  # 從 JSON 檔案載入多邊形區域配置
    zones = [
        sv.PolygonZone(
            polygon=polygon,                           # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 x、y 座標
            triggering_anchors=(sv.Position.CENTER,),  # 位置列表，指定在決定檢測是否通過線計數器時要考慮的檢測邊界框的錨點。預設情況下，這包含檢測邊界框的四個角
        )  # 建立類別用於在影格內定義多邊形區域以偵測物件
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(round(fps)) for _ in zones]  # 對每個區域使用指定的 fps 初始化 FPSBasedTimer 物件

    ocv = False      # 先結束原來程式
    time.sleep(0.5)  # 設定間隔
    ocv = True       # 再開始程式

    # 建立執行緒
    p1 = threading.Thread(target=receive, args=(rtsp_url, fps,))
    p2 = threading.Thread(target=display, args=(device, confidence, iou, classes,
                                                model, tracker, zones, timers, START, END,))
    # 啟用執行緒
    p1.start()
    p2.start()

    # 主程式結束釋放資源
    cap.release()
    print("End Start\n")


def stop():
    """ 停止主程式 """
    global ocv
    ocv = False


# 引數
zone_configuration_path = "config.json"
rtsp_url = "rtsp://localhost:8554/s"
weights = "best.pt"
device = "cuda"
confidence = 0.3
iou = 0.7
classes = []

# 放入按鈕 1 並設定參數
btn1 = QtWidgets.QPushButton(Form)
btn1.setText('start')
btn1.setFont(font)
btn1.setGeometry(50, 60, 100, 60)
btn1.clicked.connect(start)  # 點擊按鈕執行 start 函式

# 放入按鈕 2 並設定參數
btn2 = QtWidgets.QPushButton(Form)
btn2.setText('stop')
btn2.setFont(font)
btn2.setGeometry(160, 60, 100, 60)
btn2.clicked.connect(stop)  # 點擊按鈕執行 end 函式

Form.show()             # 顯示元件
sys.exit(app.exec_())   # 視窗程式結束

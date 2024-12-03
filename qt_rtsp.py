import sys
import time

from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *

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
import copy

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])  # 建立調色盤
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)  # 建立顏色註解器
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)  # 建立標籤註解器

ocv = True     # 是否執行主程式

q = queue.Queue()   # 建立佇列存放影格
q1 = queue.Queue()
q2 = queue.Queue()
start_pos = {}
end_pos = {}
recorded = {}
cur_id = {}

cur_time = 0   # 經過時間
frame_pos = 0  # 經過影格數
crop_cnt = 0   # 已截圖數
crop_max = 0   # 截圖最大數

cur_frame = -1
cur_time2 = 0

# 建立字典來追蹤越界的對象
car_crossed = {}
moto_crossed = {}

# 建立字典追蹤迴轉的車輛
wup = {}
wrongway = []

# 建立字典追蹤車輛位於線段哪一側
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
    """ 轉換 [x1 y1 x2 y2] 為 [x y w h] 格式。 """
    x_min, y_min, x_max, y_max = box  # 左上角 (x_min, y_min) 右下角 (x_max, y_max)
    x_center = (x_min + x_max) / 2  # 中心點 x 座標
    y_center = (y_min + y_max) / 2  # 中心點 y 座標
    w = x_max - x_min  # 寬度
    h = y_max - y_min  # 高度
    return [int(x_center), int(y_center), w, h]


def img_crop(frame, xx1, yy1, ww, hh, zoom) -> np.ndarray:
    """ 以倍率截圖 (xywh 格式) """
    x1 = int(xx1 - ww * (zoom - 1) / 2)
    y1 = int(yy1 - hh * (zoom - 1) / 2)
    w = int(ww * zoom)
    h = int(hh * zoom)
    return frame[y1:y1 + h, x1:x1 + w]


def find_point_side(x1, y1, x2, y2, cx, cy) -> bool:
    """ 判斷點位於線段的哪一側 """
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
        """ 自適應視窗尺寸 """
        super().resizeEvent(event)
        self.window_w, self.window_h = event.size().width(), event.size().height()
        self.video.setGeometry(self.video_x, self.video_y, self.window_w, self.window_h)

    def closeEvent(self, event):
        """ 關閉視窗結束執行主程式 """
        global ocv
        ocv = False

    @staticmethod
    def receive(rtsp_url: str, fps: float) -> None:
        """ 讀取影格 """
        global frame_pos, cur_time, ocv
        print("\nStart receive")

        cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
        ret, frame = cap.read()  # 讀取首個影格
        q.put(frame)  # 將影格放入佇列
        while ret and ocv:  # 當串流進行中
            cur_time = round(frame_pos / fps, 1)  # 計算經過秒數
            frame_pos += 1  # 計算經過影格數
            ret = cap.grab()  # 從視訊檔案或攝影機抓取下一影格，並在成功的情況下回傳 True
            if frame_pos % 2 == 0:  # 跳幀處理
                ret, image = cap.retrieve()  # 解碼並回傳下一個影格
                q.put(image)  # 將影格放入佇列

        # 串流結束釋放資源
        cap.release()
        print("End receive")

    def display(self, device: str, confidence: float, iou: float, classes: List[int],
                model: YOLO, tracker: any, zones: list, timers: list, fps: float) -> None:
        """ 接收影格處理後顯示 """
        global cur_time, frame_pos, crop_cnt, crop_max, ocv, vertex, types, START, END, area1, area2, cur_frame, cur_time2
        print("Start display")

        while ocv:  # 開始主程式
            if not q.empty():  # 若佇列不為空
                cur_frame += 1
                cur_time2 = round(cur_frame / fps * 2, 1)
                frame = q.get()  # 從佇列取出影格
                results = model(frame, verbose=False, device=device, conf=confidence)[0]  # 使用 YOLOv8 推理
                detections = sv.Detections.from_ultralytics(results)  # 根據 YOLOv8 推理結果建立檢測實例
                detections = detections[find_in_list(detections.class_id, classes)]  # 選擇僅屬於選定類別集的偵測
                detections = detections.with_nms(threshold=iou)  # 對檢測集執行非極大值抑制
                detections = tracker.update_with_detections(detections)  # 使用提供的偵測更新追蹤器並回傳更新的偵測結果

                # 建立此影格的副本
                annotated_frame = frame.copy()
                annotated_frame2 = frame.copy()

                # 取得物件邊界框和軌跡 ID
                boxes = detections.xyxy
                track_ids = detections.tracker_id
                track_cls = detections.class_id

                # 燈號檢測
                light_type = "unknown"
                for idx, cls in enumerate(types):
                    if cls == 1:
                        x1 = vertex[idx][0][0].item()
                        y1 = vertex[idx][0][1].item()
                        w = vertex[idx][2][0].item() - vertex[idx][0][0].item()
                        h = vertex[idx][2][1].item() - vertex[idx][0][1].item()
                        light_img = img_crop(frame, x1, y1, w, h, 1)
                        light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)
                        light_type = estimate_label(light_img, cur_frame, False)

                # 逐一偵測影格中物件
                if track_ids.size != 0:
                    for box, id, cls in zip(boxes, track_ids, track_cls):
                        if id not in start_pos:
                            start_pos[id] = cur_frame - 1

                        x1, y1, x2, y2 = box.astype(int)
                        cx, cy, w, h = xyxy_to_xywh(box)
                        cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)

                        id_pos = [x1, y1, x2, y2]
                        if id not in cur_id:
                            cur_id[id] = []
                        cur_id[id].append([id_pos, cur_frame])

                        # 偵測物件是否在 area1 中
                        if area1 is not None:
                            result = cv2.pointPolygonTest(area1, (cx, cy), False)
                            if result >= 0:
                                wup[id] = (cx, cy)

                        # 如果物件已經在 area1，檢查是否進入 area2
                        if id in wup and area2 is not None:
                            result1 = cv2.pointPolygonTest(area2, (cx, cy), False)
                            if result1 >= 0:
                                # 標註違規物件
                                cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 0), -1)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                                cv2.putText(annotated_frame, f'ID:{id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                if id not in wrongway:
                                    wrongway.append(id)

                        # 檢查物件是否越線（原始功能）
                        if (START.x < cx < END.x or START.x > cx > END.x) or (START.y < cy < END.y or START.y > cy > END.y):
                            cur_side[id] = find_point_side(START.x, START.y, END.x, END.y, cx, cy)
                            if id in pre_side and cur_side[id] != pre_side[id]:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                                if id not in car_crossed and cls == 1:
                                    car_crossed[id] = [True, cur_frame, 0]

                            pre_side[id] = cur_side[id]
                        # 離開線段範圍則不計入
                        elif id in pre_side:
                            pre_side.pop(id)

                # 繪製線段
                if START and END:
                    cv2.line(img=annotated_frame, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)
                    cv2.line(img=annotated_frame2, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)

                q2.put(copy.deepcopy(annotated_frame2))

                # 繪製違停區域及標註違規車輛
                for idx, zone in enumerate(zones):  # 將可迭代的對象（如列表、元組或字串）轉換為索引序列，同時列出資料和資料對應的索引值 (idx)
                    annotated_frame = sv.draw_polygon(
                        scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                    )  # 在場景上繪製多邊形

                    detections_in_zone = detections[zone.trigger(detections)]  # 將檢測與 PolygonZone 結合使用來清除區域內外的邊界框
                    time_in_zone = timers[idx].tick(detections_in_zone)  # 處理目前影格，更新每個追蹤器的持續時間
                    custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)
                    # 傳回給定形狀和類型的新陣列，並用 idx 填滿，用於定義將顏色對應到註解的策略的枚舉類別

                    annotated_frame = COLOR_ANNOTATOR.annotate(
                        scene=annotated_frame,
                        detections=detections_in_zone,
                        custom_color_lookup=custom_color_lookup,
                    )  # 用顏色註解場景中區域

                    # 建立標籤 (將id.時間結合) (跳格處理因此時間需x2)
                    labels = []
                    for id, time in zip(detections_in_zone.tracker_id, time_in_zone):
                        labels.append(f"#{id} {int(time * 2 // 60):02d}:{int((time * 2 % 60)):02d}")
                        if time * 2 > 10 and id not in end_pos:
                            end_pos[id] = cur_frame

                    annotated_frame = LABEL_ANNOTATOR.annotate(
                        scene=annotated_frame,
                        detections=detections_in_zone,
                        labels=labels,
                        custom_color_lookup=custom_color_lookup,
                    )  # 用標籤註解場景中區域

                # 繪製違規迴轉區域
                if area1 is not None:
                    cv2.polylines(annotated_frame, [area1], True, (255, 255, 255), 2)
                if area2 is not None:
                    cv2.polylines(annotated_frame, [area2], True, (255, 255, 255), 2)

                # 顯示統計資訊於畫面
                cv2.putText(annotated_frame, f"Wrong-way Cars: {len(wrongway)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Car crossed: {len(car_crossed)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Moto crossed: {len(moto_crossed)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Time: {cur_time2}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Light: {light_type}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255), 2)

                q1.put(copy.deepcopy(annotated_frame))

                # 違停錄影
                if end_pos:
                    for id, pos in end_pos.items():
                        if id not in recorded:
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片的格式為 mp4
                            out = cv2.VideoWriter(f"{cur_time}_{id}.mp4", fourcc, fps * 0.5, (1920, 1080))  # 產生空的影片
                            q1_list = list(q1.queue)
                            start = start_pos[id]
                            end = pos
                            for i in range(start, end):
                                frame = q1_list[i]
                                out.write(frame)
                            recorded[id] = True
                            out.release()

                # 越線截圖
                if car_crossed:
                    for id in list(car_crossed):
                        q2_list = list(q2.queue)
                        img_list = []
                        car_crossed[id][2] += 1
                        delay = 15
                        if car_crossed[id][2] == delay:
                            for i in [int(delay / -1.5 * 3), int(delay / -1.5 * 2), int(delay / -1.5), 0]:
                                pos = cur_frame + i
                                if 0 <= pos <= len(q2_list):
                                    index_map = {item[1]: idx for idx, item in enumerate(cur_id[id])}
                                    index = index_map.get(pos)
                                    if index is not None:
                                        x1, y1, x2, y2 = cur_id[id][index][0]
                                        cv2.rectangle(q2_list[pos], (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        cv2.putText(q2_list[pos], f'ID:{id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                                    cv2.putText(q2_list[pos], f"{len(img_list) + 1}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                                    img_list.append(q2_list[pos])
                                    if len(img_list) == 2:
                                        img_hor1 = cv2.hconcat([img_list[0], img_list[1]])
                                    elif len(img_list) == 4:
                                        img_hor2 = cv2.hconcat([img_list[2], img_list[3]])
                                        img_ver = cv2.vconcat([img_hor1, img_hor2])
                                        filename = "save/" + str(cur_frame) + '_id(' + str(id) + ').jpg'
                                        cv2.imwrite(filename, img_ver)
                            car_crossed.pop(id)

                # 顯示統計資訊於視窗
                self.label2.setText(f"Total crossed: {len(car_crossed) + len(moto_crossed)}")
                self.label3.setText(f"Time: {cur_time2}s")
                self.label4.setText(f"Light: {light_type}")

                frame = cv2.resize(annotated_frame, (1280, 720))  # 改變影像尺寸
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 影像轉換成 RGB
                height, width, channel = frame.shape  # 讀取影像尺寸和 channel 數量
                bytesPerline = channel * width  # 設定 bytesPerline (用於轉換)
                img = QImage(frame, width, height, bytesPerline, QImage.Format_RGB888)  # 轉換影像為 QImage，讓 PyQt5 可以讀取
                self.video.setPixmap(QPixmap.fromImage(img))  # label 顯示影像

        print("End display")

    def start(self):
        """ 執行主程式 """
        global ocv, vertex, types, START, END, area1, area2
        print("Start start")

        model = YOLO(self.weights)  # 初始化 YOLO 模型
        cap = cv2.VideoCapture(self.rtsp_url)  # 擷取串流影像
        fps = cap.get(cv2.CAP_PROP_FPS)  # 取得串流之 fps
        tracker = sv.ByteTrack(frame_rate=round(fps), track_activation_threshold=self.confidence)  # 初始化 ByteTrack 物件

        vertex = load_zones_config(file_path=self.zone_configuration_path)  # 從 JSON 檔案載入多邊形區域配置
        types = load_zones_config(file_path=self.type_configuration_path)   # 1 紅綠燈, 2 車流線, 3 迴轉區, 4 臨停區

        # 定義違停區
        polygons = []
        for idx, cls in enumerate(types):
            if cls == 4:
                polygons.append(vertex[idx])

        zones = [
            sv.PolygonZone(
                polygon=polygon,  # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 x、y 座標
                triggering_anchors=(sv.Position.CENTER,),  # 位置列表，指定在決定檢測是否通過線計數器時要考慮的檢測邊界框的錨點。預設情況下，這包含檢測邊界框的四個角
            )  # 建立類別用於在影格內定義多邊形區域以偵測物件
            for polygon in polygons
        ]
        timers = [FPSBasedTimer(round(fps)) for _ in zones]  # 對每個區域使用指定的 fps 初始化 FPSBasedTimer 物件

        # 定義線段
        for idx, cls in enumerate(types):
            if cls == 2:
                START = sv.Point(vertex[idx][0][0].item(), vertex[idx][0][1].item())
                END = sv.Point(vertex[idx][1][0].item(), vertex[idx][1][1].item())

        # 新增區域 area1 和 area2，用於檢測違規迴轉行為
        cnt = 1
        for idx, cls in enumerate(types):
            if cls == 3:  # 假設在配置中類別為 3 的是 area1 下一個是 area2
                if cnt % 2:
                    area1 = np.array(vertex[idx], np.int32)
                else:
                    area2 = np.array(vertex[idx], np.int32)
                cnt += 1

        ocv = False  # 先結束原來程式
        time.sleep(0.1)  # 設定間隔
        ocv = True  # 再開始程式

        # 建立執行緒
        p1 = threading.Thread(target=self.receive, args=(self.rtsp_url, fps,))
        p2 = threading.Thread(target=self.display, args=(self.device, self.confidence, self.iou, self.classes,
                                                         model, tracker, zones, timers, fps,))
        # 啟用執行緒
        p1.start()
        p2.start()

        # 主程式結束釋放資源
        cap.release()
        print("End start\n")

    @staticmethod
    def stop():
        """ 停止主程式 """
        global ocv
        ocv = False
        print("stop\n")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

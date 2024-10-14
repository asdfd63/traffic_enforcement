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

# 說明：串流，https://blog.csdn.net/submarineas/article/details/110083906

q = queue.Queue()  # 建立佇列存放影格
release = False  # 是否釋放資源
is_ret = True  # 串流是否結束

cur_time = 0  # 經過時間
crop_cnt = 0  # 已截圖數
crop_max = 0  # 截圖最大數
frame_cnt = 0  # 總影格數

# 建立字典來追蹤越界的對象
crossed_objects = {}
crossed_objects_moto = {}

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
    print("Start receive")
    cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
    ret, frame = cap.read()  # 讀取首個影格
    q.put(frame)  # 將影格放入佇列
    fp = 0  # 經過影格數
    global release, is_ret, frame_cnt, cur_time
    while ret:  # 當串流進行中
        frame_cnt = frame_cnt + 1
        cur_time = round(frame_cnt / fps, 1)  # 計算經過時間
        fp += 1
        ret = cap.grab()  # 從視訊檔案或攝影機抓取下一影格，並在成功的情況下回傳 true
        if fp % 2 == 0:  # 跳幀處理
            ret, image = cap.retrieve()  # 解碼並回傳下一個影格
            q.put(image)  # 將影格放入佇列
        if release:
            cap.release()  # 釋放資源
    cap.release()
    is_ret = False


def display(zone_configuration_path: str, device: str, confidence: float, iou: float, classes: List[int],
            model: YOLO, tracker: any, zones: list, timers: list) -> None:
    """ 接收影格處理後顯示 """

    print("Start display")
    global frame_cnt, crop_cnt, cur_time, crop_max

    # 定義線段座標 (停止線)
    points = load_zones_config(file_path=zone_configuration_path)
    START = sv.Point(points[0][0][0].item(), points[0][0][1].item())
    END = sv.Point(points[0][1][0].item(), points[0][1][1].item())
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    while True:
        if not q.empty():  # 若佇列不為空
            frame = q.get()  # 從佇列取出影格
            results = model(frame, verbose=False, device=device, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(results)  # 根據 YOLOv8 推理結果建立檢測實例
            detections = detections[find_in_list(detections.class_id, classes)]  # 選擇僅屬於選定類別集的偵測
            detections = detections.with_nms(threshold=iou)  # 對檢測集執行非極大值抑制
            detections = tracker.update_with_detections(detections)  # 使用提供的偵測更新追蹤器並回傳更新的偵測結果

            # 建立此影格的副本
            annotated_frame = frame.copy()

            # 取得邊界框和軌跡 ID
            boxes = detections.xyxy
            track_ids = detections.tracker_id
            track_cls = detections.class_id

            # 燈號檢測
            light_img = img_crop(frame, 1345, 311, 70, 19, 1)
            light_img = cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB)
            light_type = estimate_label(light_img, frame_cnt, False)

            # 繪製軌跡並計算越線物體的數量
            for box, track_id, track_clss in zip(boxes, track_ids, track_cls):
                x, y, w, h = xyxy_to_xywh(box)
                x1, y1, x2, y2 = box

                # 檢查物體是否越線
                if light_type == "red" and START.x < x < END.x and abs(y - START.y) < 5:  # 當紅燈下物體水平交叉
                    # 當物件越過線時對其進行註釋
                    cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (0, 255, 0), 1)

                    if track_id not in crossed_objects:
                        if track_clss == 1:  # 0:bus 1:car 2:moto
                            crossed_objects[track_id] = True
                            if crop_cnt < crop_max:  # 截圖
                                crop_img = img_crop(annotated_frame, x1, y1, h, w, zoom=1.5)
                                filename = "save/" + str(cur_time) + 's_ID(' + str(track_id) + ').jpg'
                                cv2.imwrite(filename, crop_img)
                                crop_cnt += 1

                    if track_id not in crossed_objects_moto:
                        if track_clss == 2:  # 0:bus 1:car 2:moto
                            crossed_objects_moto[track_id] = True
                            if crop_cnt < crop_max:  # 截圖
                                crop_img = img_crop(annotated_frame, x1, y1, w, h, zoom=3)
                                filename = "save/" + str(cur_time) + 's_ID(' + str(track_id) + ').jpg'
                                cv2.imwrite(filename, crop_img)
                                crop_cnt += 1

            # 繪製停止線於視窗
            cv2.line(img=annotated_frame, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)

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
                labels = [
                    f"#{tracker_id} {int(time * 2 // 60):02d}:{int((time * 2 % 60)):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]  # 建立標籤 (將id.時間結合) (跳格處理因此時間需乘二)
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )  # 用標籤註解場景中區域

            # 場景加入文字
            count_text = f"Objects crossed car: {len(crossed_objects)}"
            count_text_moto = f"Objects crossed moto: {len(crossed_objects_moto)}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(annotated_frame, count_text_moto, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(annotated_frame, str(cur_time), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            cv2.putText(annotated_frame, str(light_type), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("frame", annotated_frame)  # 顯示場景於視窗

        if cv2.waitKey(1) & 0xFF == ord('q') or not is_ret:  # 當按下 Q 或是串流已結束
            global release
            release = True  # 需釋放資源
            break


def main(zone_configuration_path: str,
         rtsp_url: str,
         weights: str,
         device: str,
         confidence: float,
         iou: float,
         classes: List[int],) -> None:

    model = YOLO(weights)  # 初始化 YOLO 模型
    cap = cv2.VideoCapture(rtsp_url)  # 擷取串流影像
    fps = cap.get(cv2.CAP_PROP_FPS)  # 取得串流之 fps
    tracker = sv.ByteTrack(frame_rate=round(fps), track_activation_threshold=confidence)  # 初始化 ByteTrack 物件

    polygons = load_zones_config(file_path=zone_configuration_path)  # 從 JSON 檔案載入多邊形區域配置
    zones = [
        sv.PolygonZone(
            polygon=polygon,  # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 x、y 座標
            triggering_anchors=(sv.Position.CENTER,),  # 位置列表，指定在決定檢測是否通過線計數器時要考慮的檢測邊界框的錨點。預設情況下，這包含檢測邊界框的四個角
        )  # 建立類別用於在影格內定義多邊形區域以偵測物件
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(round(fps)) for _ in zones]  # 對每個區域使用指定的 fps 初始化 FPSBasedTimer 物件

    p1 = threading.Thread(target=receive, args=(rtsp_url, fps,))
    p2 = threading.Thread(target=display, args=(zone_configuration_path, device, confidence, iou, classes,
                                                model, tracker, zones, timers,))
    p1.start()
    p2.start()


if __name__ == "__main__":  # 當此程式是被直接執行而非被引用
    # 命令列解析教學：https://haosquare.com/python-argparse/
    parser = argparse.ArgumentParser(
        description="using RTSP stream."
        # 敘述：此程式使用串流
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="config.json",
        help="Path to the zone configuration JSON file.",
    )  # 引數1：區域配置JSON檔案路徑
    parser.add_argument(
        "--rtsp_url",
        type=str,
        default="rtsp://localhost:8554/s",
        help="Complete RTSP URL for the video stream.",
    )  # 引數2：來源串流網址
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )  # 引數3：模型路徑
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )  # 引數4：執行裝置
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )  # 引數5：置信度閾值
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.7,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )  # 引數6：交並比閾值
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )  # 引數7：偵測類別 (預設全部)
    args = parser.parse_args()  # 解析命令列引數並獲取解析結果

    main(
        zone_configuration_path=args.zone_configuration_path,
        rtsp_url=args.rtsp_url,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )  # 將解析結果存入變數後引入主程式

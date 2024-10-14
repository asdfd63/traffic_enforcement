import argparse
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv
from detect import estimate_label


def xyxy_to_xywh(box):
    """ 轉換 [x1 y1 x2 y2] 格式為 [x y w h] 格式。 """
    x_min, y_min, x_max, y_max = box  # 左上角 (x_min, y_min) 右下角 (x_max, y_max)
    x_center = (x_min + x_max) / 2  # 中心點 x 座標
    y_center = (y_min + y_max) / 2  # 中心點 y 座標
    w = x_max - x_min  # 寬度
    h = y_max - y_min  # 高度
    return [x_center, y_center, w, h]


def img_crop(frame, xx1, yy1, ww, hh, zoom):
    """ 截圖 """
    x1 = int(xx1 - ww * (zoom - 1) / 2)
    y1 = int(yy1 - hh * (zoom - 1) / 2)
    w = int(ww * zoom)
    h = int(hh * zoom)
    return frame[y1:y1 + h, x1:x1 + w]


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])  # 建立調色盤(4種顏色)
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)  # 建立顏色註解器
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)  # 建立標籤註解器


def main(source_video_path: str,
         zone_configuration_path: str,
         weights: str,
         device: str,
         confidence: float,
         iou: float,
         classes: List[int],) -> None:

    model = YOLO(weights)  # 初始化 YOLO 模型
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)    # 取得視訊資訊(寬.高.fps.總影格數)
    tracker = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence)  # 初始化 ByteTrack 物件
    frames_generator = sv.get_video_frames_generator(source_video_path)  # 取得一個產生視訊影格的生成器

    ''' 區域檢測 '''
    polygons = load_zones_config(file_path=zone_configuration_path)  # 從 JSON 檔案載入多邊形區域配置
    zones = [
        sv.PolygonZone(
            polygon=polygon,  # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 x、y 座標
            triggering_anchors=(sv.Position.CENTER,),  # 位置列表，指定在決定檢測是否通過線計數器時要考慮的檢測邊界框的錨點。預設情況下，這包含檢測邊界框的四個角
        )  # 創建類別用於在影格內定義多邊形區域以偵測物件
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]  # 對每個區域使用指定的 fps 初始化 FPSBasedTimer 物件

    ''' 線段檢測 '''
    # 定義線段座標
    points = load_zones_config(file_path=zone_configuration_path)
    START = sv.Point(points[0][0][0].item(), points[0][0][1].item())
    END = sv.Point(points[0][1][0].item(), points[0][1][1].item())

    # 建立字典來追蹤越界的對象
    crossed_objects = {}
    crossed_objects_moto = {}

    img_cnt = 0  # 已截圖數
    frame_cnt = 0  # 總影格數

    for frame in frames_generator:  # 提取影格
        results = model(frame, verbose=False, device=device, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(results)  # 根據 YOLOv8 推理結果建立檢測實例
        detections = detections[find_in_list(detections.class_id, classes)]  # 選擇僅屬於選定類別集的偵測
        detections = detections.with_nms(threshold=iou)  # 對檢測集執行非極大值抑制
        detections = tracker.update_with_detections(detections)  # 使用提供的偵測更新追蹤器並傳回更新的偵測結果

        # 計時
        frame_cnt = frame_cnt + 1
        cur_time = round(frame_cnt / video_info.fps, 1)

        # 創建此影格的副本
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
            if light_type == "red" and START.x < x < END.x and abs(y - START.y) < 5:  # 假設物體水平交叉
                # 當物件越過線時對其進行註釋
                cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                              (0, 255, 0), 1)

                if track_id not in crossed_objects:
                    if track_clss == 1:  # 0:bus 1:car 2:moto
                        crossed_objects[track_id] = True
                        if img_cnt < 0:  # 截圖
                            crop_img = img_crop(annotated_frame, x1, y1, h, w, zoom=1.5)
                            filename = "save/" + str(cur_time) + 's_ID(' + str(track_id) + ').jpg'
                            cv2.imwrite(filename, crop_img)
                            img_cnt += 1

                if track_id not in crossed_objects_moto:
                    if track_clss == 2:  # 0:bus 1:car 2:moto
                        crossed_objects_moto[track_id] = True
                        if img_cnt < 0:  # 截圖
                            crop_img = img_crop(annotated_frame, x1, y1, w, h, zoom=3)
                            filename = "save/" + str(cur_time) + 's_ID(' + str(track_id) + ').jpg'
                            cv2.imwrite(filename, crop_img)
                            img_cnt += 1

        cv2.line(img=annotated_frame, pt1=(START.x, START.y), pt2=(END.x, END.y), color=(0, 0, 255), thickness=2)

        for idx, zone in enumerate(zones):  # 將可迭代的對象（如列表、元組或字串）轉換為索引序列，同時列出資料和資料對應的索引值(idx)
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
                f"#{tracker_id} {int(time // 60):02d}:{int((time % 60)):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]  # 建立標籤 (將id.時間結合)
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )  # 用標籤註解場景中區域

        count_text = f"Objects crossed car: {len(crossed_objects)}"
        count_text_moto = f"Objects crossed moto: {len(crossed_objects_moto)}"
        cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, count_text_moto, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, str(cur_time), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, str(light_type), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        '''
        if light_type == "red":
            cv2.putText(annotated_frame, str(light_type), (1250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        elif light_type == "yellow":
            cv2.putText(annotated_frame, str(light_type), (1250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
        else:
            cv2.putText(annotated_frame, str(light_type), (1250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,255,0), 5)
        '''

        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":  # 當此程式是被直接執行而非被引用
    # 命令列解析教學：https://haosquare.com/python-argparse/
    parser = argparse.ArgumentParser(
        description="using video file."
        # 敘述：此程式使用視訊檔
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="config.json",
        help="Path to the zone configuration JSON file.",
    )  # 引數1：區域配置JSON檔案路徑
    parser.add_argument(
        "--source_video_path",
        type=str,
        default="video/park3.mp4",
        help="Path to the source video file.",
    )  # 引數2：來源視訊路徑
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
        default=0.7,
        type=float,
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
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )  # 將解析結果存入變數後引入主程式

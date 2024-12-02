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
import copy


def xyxy_to_xywh(box):
    """ 轉換 [x1 y1 x2 y2] 格式為 [x y w h] 格式。 """
    x_min, y_min, x_max, y_max = box  # 左上角 (x_min, y_min) 右下角 (x_max, y_max)
    x_center = (x_min + x_max) / 2  # 中心點 x 座標
    y_center = (y_min + y_max) / 2  # 中心點 y 座標
    w = x_max - x_min  # 寬度
    h = y_max - y_min  # 高度
    return [int(x_center), int(y_center), w, h]


def img_crop(frame, xx1, yy1, ww, hh, zoom):
    """ 截圖 """
    x1 = int(xx1 - ww * (zoom - 1) / 2)
    y1 = int(yy1 - hh * (zoom - 1) / 2)
    w = int(ww * zoom)
    h = int(hh * zoom)
    return frame[y1:y1 + h, x1:x1 + w]


def find_point_side(x1, y1, x2, y2, cx, cy):
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


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])  # 建立調色盤(4種顏色)
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)  # 建立顏色註解器
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)  # 建立標籤註解器


def main(source_video_path: str,
         zone_configuration_path: str,
         type_configuration_path: str,
         weights: str,
         device: str,
         confidence: float,
         iou: float,
         classes: List[int],) -> None:

    model = YOLO(weights)  # 初始化 YOLOv8 模型
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)    # 取得視訊資訊
    tracker = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=confidence)  # 初始化 ByteTrack 物件
    frames_generator = sv.get_video_frames_generator(source_video_path)  # 取得一個產生視訊影格的生成器

    width = video_info.width  # 取得影像寬度
    height = video_info.height  # 取得影像高度
    fps = video_info.fps  # 取得影像 fps
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定影片的格式為 mp4

    q = queue.Queue()
    q2 = queue.Queue()
    start_pos = {}
    end_pos = {}
    recorded = {}

    cur_id = {}

    # 建立字典追蹤越線的車輛
    car_crossed = {}
    moto_crossed = {}

    # 建立字典追蹤迴轉的車輛
    wup = {}
    wrongway = []

    # 建立字典追蹤車輛位於線段哪一側
    pre_side = {}
    cur_side = {}

    cur_frame = -1  # 經過影格數
    crop_cnt = 0  # 已截圖數
    crop_max = 0  # 截圖最大數

    vertex = load_zones_config(file_path=zone_configuration_path)  # 從 JSON 檔案載入座標配置
    types = load_zones_config(file_path=type_configuration_path)  # 1 紅綠燈, 2 車流線, 3 迴轉區, 4 臨停區

    # 定義違停區
    polygons = []
    for idx, cls in enumerate(types):
        if cls == 4:
            polygons.append(vertex[idx])

    zones = [
        sv.PolygonZone(
            polygon=polygon,  # 由形狀 (N, 2) 的 numpy 陣列表示的多邊形，包含點的 x、y 座標
            triggering_anchors=(sv.Position.CENTER,),  # 使用中心點作為觸發點
        )  # 建立類別用於在影格內定義多邊形區域以偵測物件
        for polygon in polygons
    ]
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]  # 對每個區域使用指定的 fps 初始化 FPSBasedTimer 物件

    # 定義線段
    START = None
    END = None
    for idx, cls in enumerate(types):
        if cls == 2:
            START = sv.Point(vertex[idx][0][0].item(), vertex[idx][0][1].item())
            END = sv.Point(vertex[idx][1][0].item(), vertex[idx][1][1].item())

    # 新增區域 area1 和 area2，用於檢測違規迴轉行為
    area1 = None
    area2 = None
    cnt = 1
    for idx, cls in enumerate(types):
        if cls == 3:  # 假設在配置中類別為 3 的是 area1 下一個是 area2
            if cnt % 2:
                area1 = np.array(vertex[idx], np.int32)
            else:
                area2 = np.array(vertex[idx], np.int32)
            cnt += 1

    for frame in frames_generator:  # 提取影格
        cur_time = round(cur_frame / video_info.fps, 1)
        cur_frame += 1

        results = model(frame, verbose=False, device=device, conf=confidence)[0]  # 使用 YOLOv8 推理
        detections = sv.Detections.from_ultralytics(results)  # 根據 YOLOv8 推理結果建立檢測實例
        detections = detections[find_in_list(detections.class_id, classes)]  # 選擇僅屬於選定類別集的偵測
        detections = detections.with_nms(threshold=iou)  # 對檢測集執行非極大值抑制
        detections = tracker.update_with_detections(detections)  # 使用提供的偵測更新追蹤器並傳回更新的偵測結果

        # 建立此影格的副本
        annotated_frame = frame.copy()
        annotated_frame2 = frame.copy()

        # 取得物件邊界框和軌跡ID
        boxes = detections.xyxy
        track_ids = detections.tracker_id
        track_clss = detections.class_id

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
        for box, id, cls in zip(boxes, track_ids, track_clss):
            if id not in start_pos:
                start_pos[id] = cur_frame - 1

            x1, y1, x2, y2 = box.astype(int)
            cx, cy, w, h = xyxy_to_xywh(box)
            # cv2.circle(annotated_frame, (cx, cy), 4, (255, 0, 255), -1)

            cur_pos = [x1, y1, x2, y2]
            if id not in cur_id:
                cur_id[id] = []
            cur_id[id].append([cur_pos, cur_frame])

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
                    cv2.putText(annotated_frame, f'ID:{id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
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
        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )

            labels = []
            for id, time in zip(detections_in_zone.tracker_id, time_in_zone):
                labels.append(f"#{id} {int(time // 60):02d}:{int((time % 60)):02d}")
                if time > 10 and id not in end_pos:
                    end_pos[id] = cur_frame

            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        # 繪製違規迴轉區域
        if area1 is not None:
            cv2.polylines(annotated_frame, [area1], True, (255, 255, 255), 2)
        if area2 is not None:
            cv2.polylines(annotated_frame, [area2], True, (255, 255, 255), 2)

        # 顯示統計資訊
        cv2.putText(annotated_frame, f"Wrong-way Cars: {len(wrongway)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Car crossed: {len(car_crossed)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Moto crossed: {len(moto_crossed)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Time: {cur_time}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Light: {light_type}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        q.put(annotated_frame)

        if end_pos:
            for id, pos in end_pos.items():
                if id not in recorded:
                    out = cv2.VideoWriter(f"{cur_time}_{id}.mp4", fourcc, fps, (width, height))  # 產生空的影片
                    q_list = list(q.queue)
                    start = start_pos[id]
                    end = pos
                    for i in range(start, end):
                        frame = q_list[i]
                        out.write(frame)
                    recorded[id] = True
                    out.release()

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
                                cv2.putText(q2_list[pos], f'ID:{id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                            (36, 255, 12), 2)
                            cv2.putText(q2_list[pos], f"{len(img_list) + 1}", (50, 200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0), 8)
                            img_list.append(q2_list[pos])
                            if len(img_list) == 2:
                                img_hor1 = cv2.hconcat([img_list[0], img_list[1]])
                            elif len(img_list) == 4:
                                img_hor2 = cv2.hconcat([img_list[2], img_list[3]])
                                img_ver = cv2.vconcat([img_hor1, img_hor2])
                                filename = "save/" + str(cur_frame) + '_id(' + str(id) + ').jpg'
                                cv2.imwrite(filename, img_ver)
                    car_crossed.pop(id)

        # 顯示繪製後畫面於視窗
        cv2.namedWindow("Processed Video", cv2.WINDOW_NORMAL)
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="config.json",
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--type_configuration_path",
        type=str,
        default="config2.json",
        help="Path to the zone type configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        default="video/park.mp4",
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to the model weights file. Default is 'yolov8s.pt'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Computation device ('cpu', 'mps' or 'cuda'). Default is 'cpu'.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        type_configuration_path=args.type_configuration_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )

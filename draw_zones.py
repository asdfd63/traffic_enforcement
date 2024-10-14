import argparse
import json
import os
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import supervision as sv

# 鍵碼值
KEY_ENTER = 13
KEY_NEWLINE = 10
KEY_ESCAPE = 27
KEY_QUIT = ord("q")
KEY_SAVE = ord("s")

THICKNESS = 2  # 線段寬度
COLORS = sv.ColorPalette.DEFAULT  # 使用預設調色盤之顏色
WINDOW_NAME = "Draw Zones"  # 視窗名稱
POLYGONS = [[]]  # 多邊形區域頂點座標之列表，舉例 = [[(x1, y1),(x2, y2),(x3, y3)],[(x1, y1),(x2, y2)]]

current_mouse_position: Optional[Tuple[int, int]] = None  # 當前游標位置 (初始值為 None)


def resolve_source(source_path: str) -> Optional[np.ndarray]:
    """ 解析來源 (影像或視訊) """
    if not os.path.exists(source_path):  # 檢查路徑是否不存在
        return None

    # 先嘗試讀取影像，來源為影像則回傳
    image = cv2.imread(source_path)
    if image is not None:
        return image

    # 來源非影像則為視訊
    frame_generator = sv.get_video_frames_generator(source_path=source_path)   # 取得一個產生視訊影格的生成器
    # 說明：https://supervision.roboflow.com/utils/video/#get_video_frames_generator
    frame = next(frame_generator)  # 取得首個視訊影格
    return frame  # 回傳此影格


def mouse_event(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """ 處理滑鼠事件 """
    global current_mouse_position
    if event == cv2.EVENT_MOUSEMOVE:  # 若事件為滑動
        current_mouse_position = (x, y)  # 更新當前游標位置
    elif event == cv2.EVENT_LBUTTONDOWN:  # 若事件為左鍵點擊
        POLYGONS[-1].append((x, y))  # 新增頂點座標於列表中最後一個區域
    # 說明，滑鼠事件：https://blog.51cto.com/devops2016/2084084


def redraw(image: np.ndarray, original_image: np.ndarray) -> None:
    """ 逐一繪製區域中各頂點連接之線段 """
    global POLYGONS, current_mouse_position
    image[:] = original_image.copy()  # 建立新影像
    for idx, polygon in enumerate(POLYGONS):  # 遍歷列表中的每一個區域，取得個別區域及其索引值 (新增加區域在後)
        color = (
            COLORS.by_idx(idx).as_bgr()
            if idx < len(POLYGONS) - 1
            else sv.Color.WHITE.as_bgr()
        )  # 依據索引值決定線段顏色

        if len(polygon) > 1:
            # 若區域超過一個頂點 (線段或多邊形)
            for i in range(1, len(polygon)):  # 遍歷區域的每個頂點
                cv2.line(
                    img=image,  # 影像來源
                    pt1=polygon[i - 1],  # 線段起點座標
                    pt2=polygon[i],  # 線段終點座標
                    color=color,  # 線段顏色
                    thickness=THICKNESS,  # 線段寬度
                )  # 連接前後頂點，繪製此直線於新影像
            if idx < len(POLYGONS) - 1:  # 若不是最新區域
                cv2.line(
                    img=image,
                    pt1=polygon[-1],
                    pt2=polygon[0],
                    color=color,
                    thickness=THICKNESS,
                )  # 連接頭尾頂點，繪製此直線於新影像
        if idx == len(POLYGONS) - 1 and current_mouse_position is not None and polygon:
            # 若為最新區域且其不為空 (至少包含一點)
            cv2.line(
                img=image,
                pt1=polygon[-1],
                pt2=current_mouse_position,
                color=color,
                thickness=THICKNESS,
            )  # 連接最後一點與當前游標位置，繪製此直線於新影像
    cv2.imshow(WINDOW_NAME, image)  # 顯示新影像於視窗


def save_polygons_to_json(polygons, target_path):
    """ 寫入區域座標至指定 JSON 檔 """
    data_to_save = polygons if polygons[-1] else polygons[:-1]
    # 若列表鐘最後一個多邊形為空，則移除最後一個區域賦值給 data_to_save
    with open(target_path, "w") as f:
        json.dump(data_to_save, f)  # 寫入 data_to_save 至 JSON 檔
    # 說明，JSON 操作：https://reurl.cc/E6WZzv


def main(source_path: str, zone_configuration_path: str) -> None:
    global current_mouse_position
    original_image = resolve_source(source_path=source_path)  # 儲存解析後影像
    if original_image is None:  # 檢查影像是否正確解析
        print("Failed to load source image.")
        return

    image = original_image.copy()  # 建立新影像
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)  # 讓視窗可以自由縮放大小
    cv2.imshow(WINDOW_NAME, image)  # 顯示新影像於視窗
    cv2.setMouseCallback(WINDOW_NAME, mouse_event, image)  # 設定滑鼠事件響應於視窗

    while True:
        key = cv2.waitKey(1) & 0xFF  # 等待按鍵事件
        if key == KEY_ENTER or key == KEY_NEWLINE:  # 按下 Enter
            POLYGONS.append([])  # 增加空區域於列表末尾
        elif key == KEY_ESCAPE:  # 按下 Esc
            POLYGONS[-1] = []  # 重設最新區域
            current_mouse_position = None  # 初始化游標位置
        elif key == KEY_SAVE:  # 按下 S
            save_polygons_to_json(POLYGONS, zone_configuration_path)  # 儲存區域列表至指定 JSON 檔
            break
        redraw(image, original_image)  # 顯示繪製畫面
        if key == KEY_QUIT:  # 按下 Q
            break  # 結束繪製

    cv2.destroyAllWindows()  # 關閉所有視窗


if __name__ == "__main__":  # 當此程式是被直接執行而非被引用
    # 說明，命令列解析：https://haosquare.com/python-argparse/
    parser = argparse.ArgumentParser(
        description="Interactively draw polygons on images or video frames and save "
        "the annotations."
    )
    parser.add_argument(
        "--source_path",
        type=str,
        default="video/park2.mp4",
        help="Path to the source image or video file for drawing polygons.",
    )  # 引數 1：來源路徑
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="config.json",
        help="Path where the polygon annotations will be saved as a JSON file.",
    )  # 引數 2：配置 JSON 檔路徑
    arguments = parser.parse_args()  # 解析命令列引數並獲取解析結果
    main(
        source_path=arguments.source_path,
        zone_configuration_path=arguments.zone_configuration_path,
    )  # 將解析結果存入變數後引入主程式

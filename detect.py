import cv2
import numpy as np
import matplotlib.pyplot as plt


def find_none_zero(rgb_image: np.ndarray) -> int:
    """ 計算非零像素數量 (參數：圖片) """
    height = rgb_image.shape[0]  # 圖片高度
    width = rgb_image.shape[1]  # 圖片寬度
    counter = 0
    for h in range(height):  # 遍歷所有像素
        for w in range(width):
            pixels = rgb_image[h, w]
            if sum(pixels) != 0:
                counter = counter + 1
    return counter


def estimate_label(rgb_image: np.ndarray, count: int, display: bool = False) -> str:
    """ 估計顏色 (參數 1：圖片, 參數 2：是否顯示) """
    # 使用 HSV 實驗式地的確定每張影像中的紅色、綠色和黃色部分
    # 確定顏色閾值。傳回基於數值的分類
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    sum_saturation = np.sum(hsv[:, :, 1])  # 對飽和度求和
    area = 3232
    avg_saturation = sum_saturation / area  # 計算平均飽和度

    sat_low = int(avg_saturation * 1.3)  # 設定均值的 1.3 倍為飽和度下限
    val_low = 140  # 設定亮度下限
    # 綠色
    lower_green = np.array([35, sat_low, val_low])  # 設定顏色下限 (hsv) 70
    upper_green = np.array([100, 255, 255])  # 設定顏色上限 (hsv) 100
    green_mask = cv2.inRange(hsv, lower_green, upper_green)  # 抓取特定範圍顏色建立遮罩
    # 說明：https://steam.oxxostudio.tw/category/python/ai/opencv-inrange.html
    green_result = cv2.bitwise_and(rgb_image, rgb_image, mask=green_mask)  # 將兩張圖片的像素顏色和遮罩，進行交集運算
    # 說明：https://steam.oxxostudio.tw/category/python/ai/opencv-mask.html#a2
    # 黃色
    lower_yellow = np.array([11, sat_low, val_low])  # 10
    upper_yellow = np.array([34, 255, 255])  # 60
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_result = cv2.bitwise_and(rgb_image, rgb_image, mask=yellow_mask)
    # 紅色 1
    lower_red = np.array([0, sat_low, val_low])  # 150
    upper_red = np.array([10, 255, 255])  # 180
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask=red_mask)
    # 紅色 2
    lower_red2 = np.array([150, sat_low, val_low])  # 150
    upper_red2 = np.array([180, 255, 255])  # 180
    red2_mask = cv2.inRange(hsv, lower_red2, upper_red2)
    red2_result = cv2.bitwise_and(rgb_image, rgb_image, mask=red2_mask)
    # 計算各顏色非零像素數量
    sum_green = find_none_zero(green_result)
    sum_red = find_none_zero(red_result)
    sum_yellow = find_none_zero(yellow_result)
    sum_red2 = find_none_zero(red2_result)
    if display and count % 50 == 0:
        fig, ax = plt.subplots(1, 5, figsize=(20, 10))
        ax[0].set_title('rgb image')
        ax[0].imshow(rgb_image)
        ax[1].set_title('red ' + str(sum_red+sum_red2))
        ax[1].imshow(red_result)
        ax[2].set_title('yellow ' + str(sum_yellow))
        ax[2].imshow(yellow_result)
        ax[3].set_title('green ' + str(sum_green))
        ax[3].imshow(green_result)
        ax[4].set_title('red ' + str(sum_red2))
        ax[4].imshow(red2_result)
        # ax[4].set_title('hsv image')
        # ax[4].imshow(hsv)
        # plt.show()

    # 依據相對大小估計顏色
    if sum_red + sum_red2 >= sum_yellow and sum_red + sum_red2 >= sum_green:
        color = "red"
    elif sum_yellow >= sum_green:
        color = "yellow"
    else:
        color = "green"

    if display and count % 50 == 0:
        fig.savefig("./save/" + color + str(
            int(count / 50)) + '.png')
        plt.close(fig)

    return color

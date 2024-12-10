import json
import os
import sys
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
import random

# 檔案名稱
CONFIG_JSON_FILE = 'config.json'
CONFIG2_JSON_FILE = 'config2.json'


# 初始化 JSON 檔案
def initialize_json_files():
    if not os.path.isfile(CONFIG_JSON_FILE):
        with open(CONFIG_JSON_FILE, 'w') as f:
            json.dump([], f)
    if not os.path.isfile(CONFIG2_JSON_FILE):
        with open(CONFIG2_JSON_FILE, 'w') as f:
            json.dump([], f)


# 儲存區域的座標到 config.json，類型到 config2.json
def save_coordinates_to_json(polygons, types):
    with open(CONFIG_JSON_FILE, "w") as f:
        json.dump(polygons, f)
    with open(CONFIG2_JSON_FILE, "w") as f:
        json.dump(types, f)


class VideoLabel(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super(VideoLabel, self).__init__(parent)
        self.pixmap = None
        self.polygons = [[]]
        self.colors = []
        self.types = []
        self.current_mouse_position = None
        self.hint_text = ''
        self.file_path_text = ''
        self.status_text = ''
        self.current_type = 0
        self.setMouseTracking(True)
        self.setScaledContents(True)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setStyleSheet('border: 1px solid #f5f5f5')

    def add_polygon_color(self):
        color = QtGui.QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.colors.append(color)

    def set_pixmap(self, pixmap):
        self.pixmap = pixmap
        self.setPixmap(self.pixmap)
        self.resize(self.pixmap.size())

    def set_area_type(self, area_type):
        self.current_type = area_type

    def set_hint_text(self, text):
        self.hint_text = text
        self.update_drawing()

    def set_file_path_text(self, text):
        self.file_path_text = text
        self.update_drawing()

    def set_status_text(self, text):
        self.status_text = text
        self.update_drawing()

    def map_to_original_coordinates(self, x, y):
        if not self.pixmap:
            return x, y

        current_width = self.width()
        current_height = self.height()
        original_width = self.pixmap.width()
        original_height = self.pixmap.height()

        scale_x = original_width / current_width
        scale_y = original_height / current_height

        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        return original_x, original_y

    def mousePressEvent(self, event):
        if self.current_type == 0:
            self.set_hint_text("無操作")
            return

        if event.button() == QtCore.Qt.LeftButton and self.pixmap:
            x, y = self.map_to_original_coordinates(event.x(), event.y())
            self.polygons[-1].append([x, y])
            self.current_mouse_position = None
            self.update_drawing()
        elif event.button() == QtCore.Qt.RightButton:
            self.complete_polygon()

    def mouseMoveEvent(self, event):
        if self.current_type == 0:
            self.set_hint_text("無操作")
            return

        if self.pixmap:
            x, y = self.map_to_original_coordinates(event.x(), event.y())
            self.current_mouse_position = (x, y)
            self.update_drawing()
        self.set_hint_text(f'滑鼠座標: {event.x()}, {event.y()}')

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Escape:
            self.reset_drawing()

    def reset_drawing(self):
        if self.polygons and self.polygons[-1]:  # 如果有未完成的線段，清除它
            self.polygons[-1] = []
            self.set_hint_text("已清除當前線段")
        else:
            self.set_hint_text("無線段可清除")
        self.update_drawing()

    def complete_polygon(self):
        if self.current_type == 0:
            return

        if self.polygons[-1]:
            self.types.append(self.current_type)
            self.polygons.append([])
            self.add_polygon_color()
            save_coordinates_to_json(self.polygons[:-1], self.types)
            self.set_hint_text("區域繪製完成")
            self.update_drawing()

    def update_drawing(self):
        if not self.pixmap:
            return

        temp_pixmap = self.pixmap.copy()
        painter = QPainter(temp_pixmap)

        for idx, polygon in enumerate(self.polygons):
            color = self.colors[idx] if idx < len(self.colors) else QtCore.Qt.white
            pen = QPen(color, 3, QtCore.Qt.SolidLine)
            painter.setPen(pen)

            if len(polygon) > 1:
                for i in range(1, len(polygon)):
                    start_point = QtCore.QPointF(polygon[i - 1][0], polygon[i - 1][1])
                    end_point = QtCore.QPointF(polygon[i][0], polygon[i][1])
                    painter.drawLine(start_point, end_point)

                if idx < len(self.polygons) - 1:
                    painter.drawLine(
                        QtCore.QPointF(polygon[-1][0], polygon[-1][1]),
                        QtCore.QPointF(polygon[0][0], polygon[0][1])
                    )

            if idx == len(self.polygons) - 1 and self.current_mouse_position and polygon:
                last_point = QtCore.QPointF(polygon[-1][0], polygon[-1][1])
                current_point = QtCore.QPointF(self.current_mouse_position[0], self.current_mouse_position[1])
                painter.drawLine(last_point, current_point)

        if self.file_path_text:
            painter.setPen(QtGui.QColor('yellow'))
            painter.drawText(10, 30, f"檔案位置: {self.file_path_text}")

        if self.status_text:
            painter.setPen(QtGui.QColor('yellow'))
            painter.drawText(10, 50, self.status_text)

        if self.hint_text:
            painter.setPen(QtGui.QColor('yellow'))
            painter.drawText(10, 70, self.hint_text)

        painter.end()
        self.setPixmap(temp_pixmap)

    def resizeEvent(self, event):
        super(VideoLabel, self).resizeEvent(event)
        self.update_drawing()


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('區域繪製程式')
        self.setGeometry(100, 100, 1280, 720)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.create_menu()
        self.create_combo_box()
        self.create_scrollable_video_area()
        self.setLayout(self.main_layout)

    def create_menu(self):
        self.menubar = QtWidgets.QMenuBar(self)
        self.menu_file = QtWidgets.QMenu('File')
        self.action_open = QtWidgets.QAction('Open')
        self.action_open.triggered.connect(self.select_video)
        self.menu_file.addAction(self.action_open)
        self.action_close = QtWidgets.QAction('Close')
        self.action_close.triggered.connect(QtWidgets.QApplication.instance().quit)
        self.menu_file.addAction(self.action_close)
        self.menubar.addMenu(self.menu_file)
        self.main_layout.setMenuBar(self.menubar)

    def create_combo_box(self):
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["無", "紅綠燈", "停止線", "迴轉區", "臨停區"])
        self.combo.currentIndexChanged.connect(self.update_status_message)
        self.main_layout.addWidget(self.combo)

    def create_scrollable_video_area(self):
        self.video_label = VideoLabel()
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidget(self.video_label)
        scroll_area.setWidgetResizable(True)
        self.main_layout.addWidget(scroll_area)

    def update_status_message(self):
        current_area = self.combo.currentText()
        if current_area == "無":
            self.video_label.set_status_text("當前選擇：無操作")
        else:
            self.video_label.set_status_text(f"目前正在繪製的區域類型：{current_area}")
        self.video_label.set_area_type(self.combo.currentIndex())

    def select_video(self):
        # file_path, _ = QtWidgets.QFileDialog.getOpenFileName(None, "選擇影片檔案", "", "影片檔案 (*.mp4 *.avi *.mov)")
        rtsp_url = "rtsp://localhost:554/s"
        self.video_label.set_file_path_text(rtsp_url)
        self.load_video_frame(rtsp_url)

    def load_video_frame(self, rtsp_url):
        cap = cv2.VideoCapture(rtsp_url)
        ret, frame = cap.read()
        if ret:
            height, width, _ = frame.shape
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            q_img = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.video_label.set_pixmap(pixmap)
        cap.release()


if __name__ == '__main__':
    initialize_json_files()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

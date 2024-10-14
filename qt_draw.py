import subprocess
import os

from PyQt5 import QtWidgets, QtGui, QtCore
import sys


app = QtWidgets.QApplication(sys.argv)
Form = QtWidgets.QWidget()
Form.setWindowTitle('qt_cmd')
Form.resize(800, 200)

filePath = ''


def select_video():
    global filePath
    filePath, filterType = QtWidgets.QFileDialog.getOpenFileNames()  # 選擇檔案對話視窗
    label.setText(filePath[0])


def open_draw(path):
    # 設置絕對路徑
    script_path = os.path.join(os.getcwd(), 'draw_zones.py')
    source_path = path[0]
    config_path = os.path.join(os.getcwd(), 'config.json')

    # 檢查檔案是否存在
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"找不到腳本檔案: {script_path}")
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"找不到源檔案: {source_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到配置檔案: {config_path}")

    # 獲取當前 Python 解釋器的絕對路徑
    python_executable = sys.executable

    # 定義命令和參數
    command = [
        python_executable, script_path,  # 改自 'python', script_path,
        '--source_path', source_path,
        '--zone_configuration_path', config_path
    ]

    # 使用 subprocess.Popen 執行命令
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # 獲取輸出和錯誤信息
    stdout, stderr = process.communicate()

    # 輸出結果
    print("標準輸出:\n", stdout)
    print("標準錯誤:\n", stderr)


label = QtWidgets.QLabel(Form)
label.setText('no such file or directory')
label.setStyleSheet('font-size:20px;')
label.setGeometry(50, 30, 700, 30)

btn1 = QtWidgets.QPushButton(Form)
btn1.setText('select')
btn1.setGeometry(50, 60, 50, 30)
btn1.clicked.connect(select_video)

btn2 = QtWidgets.QPushButton(Form)
btn2.setText('draw')
btn2.setGeometry(110, 60, 50, 30)
btn2.clicked.connect(lambda: open_draw(filePath))  # 使用 lambda 函式

Form.show()
sys.exit(app.exec_())

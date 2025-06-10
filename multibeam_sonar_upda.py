import sys
import numpy as np
import time
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QLabel,
                             QGridLayout, QFileDialog, QComboBox, QSlider, QGroupBox, QHBoxLayout, QSplitter,
                             QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QProgressBar, QMenu, QAction,
                             QToolBar, QStatusBar, QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox,
                             QScrollArea)
from PyQt5.QtCore import QTimer, Qt, QDateTime, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as signal
from PyQt5.QtGui import QPixmap


class MultibeamSonarSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("多波束测深显控平台 - 专业版")
        self.setGeometry(100, 100, 1200, 800)

        # 设置暗色主题
        self.apply_dark_theme()

        # 添加状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统就绪")

        # 添加系统日志 - 移到前面来
        self.system_log = []

        # 添加工具栏
        self.create_toolbar()

        # 初始化数据
        self.init_data()

        # 尝试设置matplotlib支持中文
        self.setup_matplotlib_chinese_support()

        # 创建选项卡
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 创建各个选项卡页面
        self.create_realtime_tab()
        self.create_3d_view_tab()

        self.create_analysis_tab()  # 新增数据分析选项卡
        self.create_settings_tab()  # 新增设置选项卡

        # 添加新的预览标签页
        self.tabs.addTab(self.create_3d_model_tab(), "三维情况模型的建立与求解")
        self.tabs.addTab(self.create_2d_model_tab(), "二维情况模型的建立与求解")
        self.tabs.addTab(self.create_comparison_tab(), "对比图")
        self.tabs.addTab(self.create_slope_model_tab(), "海底平面为坡面模型的建立与求解")
        # 设置定时器，模拟实时数据
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(500)  # 每500ms更新一次

        # 记录开始时间
        self.start_time = QDateTime.currentDateTime()
        # 创建时钟更新定时器
        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)  # 每秒更新一次

        # 记录系统启动
        self.add_log("系统启动")

    def apply_dark_theme(self):
        """应用暗色主题"""
        # 创建暗色调色板
        dark_palette = QPalette()

        # 设置各种颜色角色
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))

        # 修改以下颜色以改善对比度
        dark_palette.setColor(QPalette.Text, QColor(0, 0, 0))  # 黑色文本
        dark_palette.setColor(QPalette.Button, QColor(180, 180, 180))  # 浅灰色按钮
        dark_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))  # 黑色按钮文本

        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))  # 高亮文本为黑色

        # 设置禁用控件的颜色
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))

        # 应用调色板
        self.setPalette(dark_palette)

        # 样式表可以进一步自定义控件的外观
        self.setStyleSheet("""
            /* 选项卡样式 */
            QTabWidget::pane { 
                border: 1px solid #444;
                background: #3D3D3D;
            }

            QTabBar::tab {
                background: #656565;
                color: white;
                padding: 5px 10px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }

            QTabBar::tab:selected {
                background: #3D3D3D;
                border-bottom: 2px solid #0078D7;
            }

            QTabBar::tab:hover:!selected {
                background: #555;
            }

            /* 按钮样式 */
            QPushButton {
                background-color: #606060;
                color: black;
                border: 1px solid #808080;
                border-radius: 3px;
                padding: 5px;
                min-width: 80px;
            }

            QPushButton:hover {
                background-color: #707070;
                border: 1px solid #A0A0A0;
            }

            QPushButton:pressed {
                background-color: #505050;
            }

            /* 组框样式 */
            QGroupBox {
                font-weight: bold;
                border: 1px solid #808080;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: white;
            }

            /* 下拉菜单样式 */
            QComboBox {
                background-color: #505050;
                color: black;
                border: 1px solid #808080;
                padding: 3px;
                border-radius: 3px;
            }

            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #808080;
                border-left-style: solid;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }

            QComboBox QAbstractItemView {
                background-color: #505050;
                color: white;
                selection-background-color: #0078D7;
            }

            /* 菜单和状态栏样式 */
            QMenuBar {
                background-color: #404040;
                color: white;
            }

            QMenuBar::item {
                background: transparent;
                padding: 5px 10px;
            }

            QMenuBar::item:selected {
                background-color: #0078D7;
                color: white;
            }

            QMenu {
                background-color: #404040;
                color: white;
                border: 1px solid #505050;
            }

            QMenu::item:selected {
                background-color: #0078D7;
                color: white;
            }

            QStatusBar {
                background-color: #404040;
                color: white;
                border-top: 1px solid #505050;
            }

            /* 标签样式 */
            QLabel {
                color: white;
            }

            /* 列表和表格样式 */
            QListWidget, QTableWidget, QTreeWidget {
                background-color: #2D2D2D;
                border: 1px solid #505050;
                color: white;
            }

            QListWidget::item:selected, QTableWidget::item:selected, QTreeWidget::item:selected {
                background-color: #0078D7;
                color: white;
            }

            QHeaderView::section {
                background-color: #404040;
                color: white;
                padding: 4px;
                border: 1px solid #505050;
            }

            /* 工具栏样式 */
            QToolBar {
                background-color: #404040;
                border: 1px solid #505050;
                spacing: 3px;
            }

            QToolButton {
                background-color: #404040;
                color: white;
                border-radius: 3px;
                padding: 3px;
            }

            QToolButton:hover {
                background-color: #505050;
            }

            QToolButton:pressed {
                background-color: #353535;
            }
        """)

    def create_toolbar(self):
        """创建工具栏"""
        self.toolbar = QToolBar("主工具栏")
        self.addToolBar(self.toolbar)

        # 开始暂停按钮
        self.start_stop_action = QAction("暂停采集", self)
        self.start_stop_action.setStatusTip("暂停/开始数据采集")
        self.start_stop_action.triggered.connect(self.toggle_acquisition)
        self.toolbar.addAction(self.start_stop_action)

        self.toolbar.addSeparator()

        # 快速保存按钮
        save_action = QAction("快速保存", self)
        save_action.setStatusTip("快速保存当前数据")
        save_action.triggered.connect(self.quick_save)
        self.toolbar.addAction(save_action)

        # 快速加载按钮
        load_action = QAction("加载数据", self)
        load_action.setStatusTip("加载历史数据")
        load_action.triggered.connect(self.load_data)
        self.toolbar.addAction(load_action)

        self.toolbar.addSeparator()

        # 添加实时时钟显示
        self.clock_label = QLabel("00:00:00")
        self.toolbar.addWidget(self.clock_label)

        # 添加测量时间
        self.survey_time_label = QLabel("测量时长: 00:00:00")
        self.toolbar.addWidget(self.survey_time_label)

    def init_data(self):
        """初始化模拟数据"""
        # 采集状态
        self.acquisition_active = True

        # 模拟声呐位置
        self.position_x = 0
        self.position_y = 0
        self.vessel_heading = 0  # 船舶航向（角度）

        # GPS数据
        self.gps_lat = 30.0  # 起始纬度
        self.gps_lon = 120.0  # 起始经度

        # 模拟航迹数据
        self.track_x = []
        self.track_y = []

        # 模拟水深数据矩阵 (50x50)
        self.depth_data = np.zeros((50, 50))
        # 生成一些随机的海底地形
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        self.depth_data = 20 + 5 * np.sin(X) + 3 * np.cos(Y) + np.random.rand(50, 50) * 2

        # 模拟设备状态
        self.device_status = {
            "电源": "正常",
            "传感器": "正常",
            "数据链路": "正常",
            "存储系统": "正常",
            "GPS": "正常",
            "航向传感器": "正常",
            "网络连接": "正常",
            "温度": "正常",
            "湿度": "正常"
        }

        # 模拟多波束数据
        self.beam_count = 64  # 增加波束数量到64
        self.beam_data = np.random.rand(self.beam_count) * 10 + 20  # 20-30米深度范围
        self.beam_angles = np.linspace(-75, 75, self.beam_count)  # 扩大波束角度范围

        # 模拟系统参数
        self.system_params = {
            "声速": 1500.0,  # m/s
            "发射频率": 400,  # kHz
            "采样率": 10,  # Hz
            "功率": 100,  # %
            "深度测量范围": [5, 300],  # m
            "准确度": 0.1,  # m
            "分辨率": 0.05  # m
        }

        # 存储历史深度数据
        self.history_depth = []

        # 模拟声速剖面
        self.sound_velocity_profile = {
            "深度": np.linspace(0, 100, 20),
            "声速": np.linspace(1490, 1520, 20) + np.random.rand(20) * 5
        }

        # 标准水深区域数据
        self.calibration_area = None

        # 测量统计数据
        self.stats = {
            "points_collected": 0,
            "max_depth": 0,
            "min_depth": 100,
            "avg_depth": 0,
            "survey_duration": "00:00:00",
            "coverage_area": 0,
        }

    def update_data(self):
        """更新模拟数据"""
        if not self.acquisition_active:
            return

        # 模拟声呐移动
        self.position_x += 0.1
        self.position_y += 0.05 * np.sin(self.position_x)

        # 更新船舶航向
        self.vessel_heading = (np.arctan2(0.05 * np.cos(self.position_x), 0.1) * 180 / np.pi) % 360

        # 更新GPS位置
        self.gps_lat += 0.0001 * np.cos(self.vessel_heading * np.pi / 180)
        self.gps_lon += 0.0001 * np.sin(self.vessel_heading * np.pi / 180)

        # 更新航迹
        self.track_x.append(self.position_x)
        self.track_y.append(self.position_y)

        # 限制航迹长度，避免内存占用过大
        if len(self.track_x) > 500:  # 增加航迹长度
            self.track_x = self.track_x[-500:]
            self.track_y = self.track_y[-500:]

        # 更新波束数据 - 增加一些实际海底特征，如沟壑、山脊等
        base_depth = 20 + 5 * np.sin(self.position_x * 0.5) + 3 * np.cos(self.position_y * 0.5)
        # 添加沟壑效果
        trench_effect = 5 * np.exp(-0.1 * (self.beam_angles - 20 * np.sin(self.position_x * 0.2)) ** 2)
        # 随机波动
        noise = np.random.rand(self.beam_count) * 1.5

        self.beam_data = base_depth + trench_effect + noise

        # 随机添加一些"异常点"，模拟鱼群或障碍物
        if np.random.rand() > 0.9:
            anomaly_pos = int(np.random.rand() * self.beam_count * 0.8 + 0.1 * self.beam_count)
            anomaly_length = int(np.random.rand() * 5) + 2
            for i in range(max(0, anomaly_pos - anomaly_length // 2),
                           min(self.beam_count, anomaly_pos + anomaly_length // 2)):
                self.beam_data[i] -= np.random.rand() * 5 + 2

        # 存储历史深度数据
        self.history_depth.append(np.mean(self.beam_data))
        if len(self.history_depth) > 1000:
            self.history_depth = self.history_depth[-1000:]

        # 更新测量统计数据
        self.stats["points_collected"] = len(self.track_x)
        self.stats["max_depth"] = max(self.stats["max_depth"], np.max(self.beam_data))
        self.stats["min_depth"] = min(self.stats["min_depth"], np.min(self.beam_data))
        self.stats["avg_depth"] = np.mean(self.history_depth)
        track_length = 0
        if len(self.track_x) >= 2:
            track_length = np.sum(np.sqrt(np.diff(self.track_x) ** 2 + np.diff(self.track_y) ** 2))
        self.stats["coverage_area"] = track_length * 2 * 75 * np.pi / 180  # 近似航迹带宽

        # 随机模拟设备状态变化
        if np.random.rand() > 0.97:
            keys = list(self.device_status.keys())
            random_device = keys[int(np.random.rand() * len(keys))]
            old_status = self.device_status[random_device]
            self.device_status[random_device] = "警告" if self.device_status[random_device] == "正常" else "正常"
            if old_status != self.device_status[random_device]:
                self.add_log(f"设备状态变化: {random_device} 从 {old_status} 变为 {self.device_status[random_device]}")

        # 更新声速剖面
        if np.random.rand() > 0.95:
            delta = np.random.rand(20) * 2 - 1
            self.sound_velocity_profile["声速"] += delta
            self.add_log("声速剖面已更新")

        # 更新各个显示组件
        self.update_realtime_display()
        self.update_device_status()
        self.update_stats_display()

    def create_realtime_tab(self):
        """创建实时显示选项卡"""
        tab = QWidget()
        main_layout = QVBoxLayout()

        # 创建顶部状态栏
        status_layout = QHBoxLayout()

        # GPS坐标
        self.gps_display = QLabel("GPS: 30.0000° N, 120.0000° E")
        self.gps_display.setStyleSheet("color: cyan; font-weight: bold")
        status_layout.addWidget(self.gps_display)

        # 航向
        self.heading_display = QLabel("航向: 0.0°")
        self.heading_display.setStyleSheet("color: yellow; font-weight: bold")
        status_layout.addWidget(self.heading_display)

        # 深度
        self.current_depth_display = QLabel("当前深度: 20.0 m")
        self.current_depth_display.setStyleSheet("color: lime; font-weight: bold")
        status_layout.addWidget(self.current_depth_display)

        main_layout.addLayout(status_layout)

        # 使用分割器让用户可以调整各个区域的大小
        splitter = QSplitter(Qt.Vertical)

        # 上半部分显示区域
        top_widget = QWidget()
        layout = QGridLayout()
        top_widget.setLayout(layout)

        # 创建波束显示区域
        self.beam_plot = pg.PlotWidget()
        self.beam_plot.setTitle("多波束水深剖面图")
        self.beam_plot.setLabel('left', '深度', 'm')
        self.beam_plot.setLabel('bottom', '波束位置', '°')
        self.beam_plot.showGrid(x=True, y=True)
        self.beam_plot.setYRange(0, 50, padding=0)
        self.beam_curve = self.beam_plot.plot(pen='g')
        # 添加参考线
        self.beam_plot.addLine(y=30, pen=pg.mkPen('r', width=1, style=Qt.DashLine))

        # 增加波束置信度显示
        self.beam_confidence = pg.PlotDataItem(pen=None, symbol='o', symbolSize=3, symbolBrush='r')
        self.beam_plot.addItem(self.beam_confidence)

        # 创建航迹显示区域
        self.track_plot = pg.PlotWidget()
        self.track_plot.setTitle("航迹")
        self.track_plot.setLabel('left', 'Y', 'm')
        self.track_plot.setLabel('bottom', 'X', 'm')
        self.track_plot.showGrid(x=True, y=True)
        self.track_curve = self.track_plot.plot(pen='b')
        # 添加当前位置指示
        self.current_pos_indicator = pg.ScatterPlotItem()
        self.track_plot.addItem(self.current_pos_indicator)

        # 添加到布局
        layout.addWidget(self.beam_plot, 0, 0)
        layout.addWidget(self.track_plot, 0, 1)

        splitter.addWidget(top_widget)

        # 创建水深图显示区
        depth_widget = QWidget()
        depth_layout = QVBoxLayout()
        depth_widget.setLayout(depth_layout)

        depth_controls = QHBoxLayout()
        depth_layout.addLayout(depth_controls)

        # 添加伪彩色控制
        color_label = QLabel("伪彩色:")
        depth_controls.addWidget(color_label)

        self.color_map_combo = QComboBox()
        self.color_map_combo.addItems(['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'turbo', 'jet'])
        self.color_map_combo.currentTextChanged.connect(self.update_colormap)
        depth_controls.addWidget(self.color_map_combo)

        # 添加对比度控制
        contrast_label = QLabel("对比度:")
        depth_controls.addWidget(contrast_label)

        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(100)
        self.contrast_slider.setValue(50)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        depth_controls.addWidget(self.contrast_slider)

        depth_controls.addStretch()

        # 添加深度范围显示
        self.depth_range_label = QLabel("深度范围: 10.0 - 30.0 m")
        depth_controls.addWidget(self.depth_range_label)

        # 水深图
        self.depth_image = pg.ImageView()
        self.depth_image.setImage(self.depth_data)
        self.depth_image.setColorMap(pg.colormap.get('viridis'))
        depth_layout.addWidget(self.depth_image)

        splitter.addWidget(depth_widget)

        # 设置分割器的初始大小
        splitter.setSizes([400, 300])

        main_layout.addWidget(splitter)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "实时显示")

    def create_3d_view_tab(self):
        """创建3D视图选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建控制面板
        control_panel = QHBoxLayout()

        # 添加视图控制
        view_group = QGroupBox("视图控制")
        view_layout = QHBoxLayout()
        view_group.setLayout(view_layout)

        # 纵向夸张系数
        exaggeration_label = QLabel("纵向夸张:")
        view_layout.addWidget(exaggeration_label)

        self.exaggeration_slider = QSlider(Qt.Horizontal)
        self.exaggeration_slider.setMinimum(1)
        self.exaggeration_slider.setMaximum(10)
        self.exaggeration_slider.setValue(2)
        self.exaggeration_slider.valueChanged.connect(self.update_3d_view)
        view_layout.addWidget(self.exaggeration_slider)

        control_panel.addWidget(view_group)

        # 添加显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()
        display_group.setLayout(display_layout)

        self.show_contours = QCheckBox("显示等高线")
        self.show_contours.setChecked(True)
        self.show_contours.stateChanged.connect(self.update_3d_view)
        display_layout.addWidget(self.show_contours)

        self.show_wireframe = QCheckBox("显示网格")
        self.show_wireframe.setChecked(False)
        self.show_wireframe.stateChanged.connect(self.update_3d_view)
        display_layout.addWidget(self.show_wireframe)

        control_panel.addWidget(display_group)

        # 添加剖面控制
        profile_group = QGroupBox("剖面控制")
        profile_layout = QVBoxLayout()
        profile_group.setLayout(profile_layout)

        self.show_profile = QCheckBox("显示纵向剖面")
        self.show_profile.setChecked(False)
        self.show_profile.stateChanged.connect(self.toggle_profile_view)
        profile_layout.addWidget(self.show_profile)

        profile_pos_layout = QHBoxLayout()
        profile_pos_label = QLabel("剖面位置:")
        profile_pos_layout.addWidget(profile_pos_label)

        self.profile_slider = QSlider(Qt.Horizontal)
        self.profile_slider.setMinimum(0)
        self.profile_slider.setMaximum(49)
        self.profile_slider.setValue(25)
        self.profile_slider.valueChanged.connect(self.update_3d_view)
        self.profile_slider.setEnabled(False)
        profile_pos_layout.addWidget(self.profile_slider)

        profile_layout.addLayout(profile_pos_layout)

        control_panel.addWidget(profile_group)

        layout.addLayout(control_panel)

        # 创建3D图形
        self.figure = Figure(figsize=(8, 6), facecolor='#2D2D2D')
        self.canvas = FigureCanvas(self.figure)
        self.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax3d = self.figure.add_subplot(111, projection='3d')
        self.ax3d.set_facecolor('#2D2D2D')

        # 初始化3D图形
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.depth_data

        # 设置合适的视角
        self.ax3d.view_init(elev=30, azim=45)

        self.surf = self.ax3d.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)
        self.ax3d.set_xlabel('X (m)', color='white')
        self.ax3d.set_ylabel('Y (m)', color='white')
        self.ax3d.set_zlabel('深度 (m)', color='white')
        self.ax3d.set_title('3D海底地形图', color='white')
        self.ax3d.tick_params(colors='white')

        # 设置Z轴方向
        self.ax3d.invert_zaxis()

        layout.addWidget(self.canvas)

        # 添加更新和导出按钮
        button_layout = QHBoxLayout()

        update_button = QPushButton("刷新3D视图")
        update_button.clicked.connect(self.update_3d_view)
        button_layout.addWidget(update_button)

        export_button = QPushButton("导出3D模型")
        export_button.clicked.connect(self.export_3d_model)
        button_layout.addWidget(export_button)

        layout.addLayout(button_layout)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "3D地形图")

        # 初始调用一次更新确保正确显示
        self.update_3d_view()

    def create_3d_model_tab(self):
        """创建三维情况模型的建立与求解选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout()
        scroll_content.setLayout(scroll_layout)

        # 使用当前工作目录
        base_path = os.path.join(os.getcwd(), "image", "三维情况模型的建立与求解")
        print(f"Checking base path: {base_path}")  # 调试信息

        # 定义需要查找的文件夹
        folders = [
            "航向坡线关系图",
            "坡度为135度时几何关系图",
            "最优方向"
        ]

        # 用于存储所有图片路径和标题
        image_paths = []

        # 遍历所有文件夹
        for folder in folders:
            folder_path = os.path.join(base_path, folder)
            print(f"Checking folder: {folder_path}")  # 调试信息

            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.png', '.jpg', '.PNG', '.JPG')):
                        full_path = os.path.join(folder_path, file)
                        print(f"Found image: {full_path}")  # 调试信息
                        image_paths.append({
                            'path': full_path,
                            'title': folder
                        })
            else:
                print(f"Folder not found: {folder_path}")  # 调试信息

        if not image_paths:
            error_label = QLabel("未找到图片文件")
            error_label.setStyleSheet("color: red")
            layout.addWidget(error_label)
            tab.setLayout(layout)
            return tab

        # 添加图片到布局
        for i, config in enumerate(image_paths):
            group = QGroupBox()
            group_layout = QVBoxLayout()

            image_label = QLabel()
            pixmap = QPixmap(config['path'])

            if pixmap.isNull():
                print(f"Failed to load image: {config['path']}")  # 调试信息
                error_label = QLabel(f"无法加载图片: {config['path']}")
                error_label.setStyleSheet("color: red")
                group_layout.addWidget(error_label)
            else:
                print(f"Successfully loaded image: {config['path']}")  # 调试信息
                scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                image_label.setPixmap(scaled_pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                group_layout.addWidget(image_label)

            title_label = QLabel(config['title'])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 12px; padding: 5px;")
            group_layout.addWidget(title_label)

            group.setLayout(group_layout)
            group.setStyleSheet("""
                QGroupBox {
                    border: 1px solid #444;
                    border-radius: 5px;
                    padding: 5px;
                    background-color: #2D2D2D;
                }
            """)

            # 每行放置两张图片
            scroll_layout.addWidget(group, i // 2, i % 2)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        tab.setLayout(layout)
        return tab

    def create_2d_model_tab(self):
        """创建二维情况模型的建立与求解选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout()
        scroll_content.setLayout(scroll_layout)
        base_path = os.path.join(os.getcwd(), "image", "二维情况模型的建立与求解")

        image_paths = []

        # 获取所有相关图片
        for folder in ["投影位置简图", "投影测线间距", "相关位置简图"]:
            folder_path = os.path.join(base_path, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(('.png', '.jpg')):
                        image_paths.append({
                            'path': os.path.join(folder_path, file),
                            'title': folder
                        })

        # 添加图片到布局
        for i, config in enumerate(image_paths):
            group = QGroupBox()
            group_layout = QVBoxLayout()

            image_label = QLabel()
            pixmap = QPixmap(config['path'])
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            group_layout.addWidget(image_label)

            title_label = QLabel(config['title'])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 12px; padding: 5px;")
            group_layout.addWidget(title_label)

            group.setLayout(group_layout)
            group.setStyleSheet(
                "QGroupBox {border: 1px solid #444; border-radius: 5px; padding: 5px; background-color: #2D2D2D;}")

            scroll_layout.addWidget(group, i // 2, i % 2)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        tab.setLayout(layout)
        return tab

    def create_comparison_tab(self):
        """创建对比图选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout()
        scroll_content.setLayout(scroll_layout)

        image_paths = []

        # 获取单波束原理和多波束原理对比图片
        comparison_path = os.path.join(os.getcwd(), "image", "对比图", "单波束原理和多波束原理对比")

        if os.path.exists(comparison_path):
            for file in os.listdir(comparison_path):
                if file.endswith(('.png', '.jpg')):
                    image_paths.append({
                        'path': os.path.join(comparison_path, file),
                        'title': "单波束原理和多波束原理对比"
                    })

        # 添加图片到布局
        for i, config in enumerate(image_paths):
            group = QGroupBox()
            group_layout = QVBoxLayout()

            image_label = QLabel()
            pixmap = QPixmap(config['path'])
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)
            image_label.setAlignment(Qt.AlignCenter)
            group_layout.addWidget(image_label)

            title_label = QLabel(config['title'])
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 12px; padding: 5px;")
            group_layout.addWidget(title_label)

            group.setLayout(group_layout)
            group.setStyleSheet(
                "QGroupBox {border: 1px solid #444; border-radius: 5px; padding: 5px; background-color: #2D2D2D;}")

            scroll_layout.addWidget(group, i // 2, i % 2)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        tab.setLayout(layout)
        return tab

    def create_slope_model_tab(self):
        """创建海底平面为坡面模型的建立与求解选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QGridLayout()
        scroll_layout.setSpacing(10)
        scroll_content.setLayout(scroll_layout)

        base_path = os.path.join(os.getcwd(), "image", "海底平面为坡面模型的建立与求解")
        print(f"Base path: {base_path}")  # 调试信息

        # 定义需要处理的文件夹及其布局位置
        folders_map = {
            "不同重叠率与测线数量关系（左）与测线位置（右）": (0, False),  # (行号, 是否需要并排)
            "较浅较深情况（左）与单独抽取的圆锥结构（右）": (1, False),
            "连接后的路径": (2, True),  # 需要并排显示
            "最优方向": (2, True)  # 与连接后的路径并排
        }

        # 处理每个文件夹
        for folder, (row, is_paired) in folders_map.items():
            folder_path = os.path.join(base_path, folder)

            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

                if files:
                    group = QGroupBox()
                    group_layout = QVBoxLayout()

                    # 处理单个文件夹中的所有图片
                    images_layout = QHBoxLayout()

                    for file in sorted(files):
                        image_path = os.path.join(folder_path, file)
                        image_label = QLabel()
                        pixmap = QPixmap(image_path)
                        if not pixmap.isNull():
                            scaled_pixmap = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                            image_label.setPixmap(scaled_pixmap)
                            image_label.setAlignment(Qt.AlignCenter)
                            images_layout.addWidget(image_label)

                    group_layout.addLayout(images_layout)

                    # 添加标题
                    title_label = QLabel(folder)
                    title_label.setAlignment(Qt.AlignCenter)
                    title_label.setStyleSheet("color: rgb(200, 200, 200); font-size: 12px; padding: 5px;")
                    group_layout.addWidget(title_label)

                    group.setLayout(group_layout)
                    group.setStyleSheet("""
                        QGroupBox {
                            border: 1px solid #444;
                            border-radius: 5px;
                            padding: 5px;
                            background-color: #2D2D2D;
                            margin-top: 5px;
                        }
                    """)

                    # 如果是需要并排的图片，占用半列
                    if is_paired:
                        scroll_layout.addWidget(group, row, 0 if folder == "连接后的路径" else 1)
                    else:
                        scroll_layout.addWidget(group, row, 0, 1, 2)  # 跨两列显示

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        tab.setLayout(layout)
        return tab

    def create_analysis_tab(self):
        """创建数据分析选项卡"""
        tab = QWidget()
        main_layout = QVBoxLayout()

        # 创建顶部控制栏
        controls = QHBoxLayout()

        # 添加分析类型选择
        analysis_label = QLabel("分析类型:")
        controls.addWidget(analysis_label)

        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "深度趋势分析",
            "地形坡度分析",
            "声速剖面分析",
            "测量精度分析"
        ])
        self.analysis_type.currentIndexChanged.connect(self.switch_analysis)
        controls.addWidget(self.analysis_type)

        # 添加时间范围选择
        time_range_label = QLabel("   时间范围:")
        controls.addWidget(time_range_label)

        self.time_range = QComboBox()
        self.time_range.addItems([
            "全部",
            "最近10分钟",
            "最近1小时",
            "自定义..."
        ])
        controls.addWidget(self.time_range)

        controls.addStretch()

        # 添加分析按钮
        analyze_button = QPushButton("执行分析")
        analyze_button.clicked.connect(self.run_analysis)
        controls.addWidget(analyze_button)

        main_layout.addLayout(controls)

        # 创建分析图表区域
        self.analysis_figure = Figure(figsize=(8, 6), facecolor='#2D2D2D')
        self.analysis_canvas = FigureCanvas(self.analysis_figure)

        # 创建多个子图用于不同类型的分析
        self.depth_trend_ax = self.analysis_figure.add_subplot(111)
        self.depth_trend_ax.set_facecolor('#2D2D2D')
        self.depth_trend_ax.tick_params(colors='white')
        self.depth_trend_ax.set_xlabel('Time', color='white')  # 使用英文
        self.depth_trend_ax.set_ylabel('Depth (m)', color='white')  # 使用英文
        self.depth_trend_ax.set_title('Depth Trend Analysis', color='white')  # 使用英文
        self.analysis_figure.tight_layout()

        main_layout.addWidget(self.analysis_canvas)

        # 添加分析结果和统计信息显示区域
        results_group = QGroupBox("分析结果")
        results_layout = QGridLayout()

        # 添加一些统计信息标签
        self.stats_labels = {}
        stats_items = [
            "平均深度", "最大深度", "最小深度", "深度标准差",
            "坡度平均值", "最大坡度", "最小坡度", "坡度标准差"
        ]

        row, col = 0, 0
        for item in stats_items:
            label = QLabel(f"{item}:")
            results_layout.addWidget(label, row, col * 2)

            value_label = QLabel("N/A")
            value_label.setStyleSheet("color: yellow")
            self.stats_labels[item] = value_label
            results_layout.addWidget(value_label, row, col * 2 + 1)

            col += 1
            if col > 3:
                col = 0
                row += 1

        results_group.setLayout(results_layout)
        main_layout.addWidget(results_group)

        tab.setLayout(main_layout)
        self.tabs.addTab(tab, "数据分析")

    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 系统参数设置
        params_group = QGroupBox("系统参数")
        params_layout = QGridLayout()

        # 添加各参数控制
        param_items = [
            ("声速 (m/s)", "声速", 1450, 1550),
            ("发射频率 (kHz)", "发射频率", 200, 600),
            ("采样率 (Hz)", "采样率", 1, 50),
            ("功率 (%)", "功率", 10, 100)
        ]

        self.param_controls = {}
        row = 0
        for label_text, key, min_val, max_val in param_items:
            # 标签
            label = QLabel(f"{label_text}:")
            params_layout.addWidget(label, row, 0)

            # 将浮点值转换为整数以便QSlider使用
            current_value = int(self.system_params[key])

            # 滑块控制
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(current_value)  # 使用整数值
            slider.valueChanged.connect(lambda val, k=key: self.update_system_param(k, val))
            params_layout.addWidget(slider, row, 1)

            # 显示当前值的标签
            value_label = QLabel(f"{self.system_params[key]}")
            params_layout.addWidget(value_label, row, 2)

            self.param_controls[key] = (slider, value_label)
            row += 1

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # 声速剖面设置
        svp_group = QGroupBox("声速剖面")
        svp_layout = QVBoxLayout()

        # 添加声速剖面图
        self.svp_figure = Figure(figsize=(5, 4), facecolor='#2D2D2D')
        self.svp_canvas = FigureCanvas(self.svp_figure)
        self.svp_ax = self.svp_figure.add_subplot(111)
        self.svp_ax.set_facecolor('#2D2D2D')
        self.svp_ax.tick_params(colors='white')
        self.svp_ax.set_xlabel('Sound Velocity (m/s)', color='white')  # 使用英文字体
        self.svp_ax.set_ylabel('Depth (m)', color='white')  # 使用英文字体
        self.svp_ax.set_title('Sound Velocity Profile', color='white')  # 使用英文字体
        self.svp_ax.invert_yaxis()  # 反转Y轴，使深度增加往下

        # 绘制声速剖面
        self.svp_line, = self.svp_ax.plot(
            self.sound_velocity_profile["声速"],
            self.sound_velocity_profile["深度"],
            'r-'
        )
        self.svp_figure.tight_layout()

        svp_layout.addWidget(self.svp_canvas)

        # 添加声速剖面控制按钮
        svp_buttons = QHBoxLayout()

        load_svp_btn = QPushButton("加载声速剖面")
        load_svp_btn.clicked.connect(self.load_svp)
        svp_buttons.addWidget(load_svp_btn)

        edit_svp_btn = QPushButton("编辑声速剖面")
        edit_svp_btn.clicked.connect(self.edit_svp)
        svp_buttons.addWidget(edit_svp_btn)

        svp_layout.addLayout(svp_buttons)
        svp_group.setLayout(svp_layout)
        layout.addWidget(svp_group)

        # 校准设置
        calib_group = QGroupBox("系统校准")
        calib_layout = QHBoxLayout()

        calib_btn = QPushButton("进行系统校准")
        calib_btn.clicked.connect(self.calibrate_system)
        calib_layout.addWidget(calib_btn)

        calib_status = QLabel("上次校准: 未校准")
        calib_layout.addWidget(calib_status)

        calib_group.setLayout(calib_layout)
        layout.addWidget(calib_group)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "设置")

    def update_realtime_display(self):
        """更新实时显示"""
        # 更新顶部状态
        self.gps_display.setText(f"GPS: {self.gps_lat:.4f}° N, {self.gps_lon:.4f}° E")
        self.heading_display.setText(f"航向: {self.vessel_heading:.1f}°")
        current_depth = np.mean(self.beam_data)
        self.current_depth_display.setText(f"当前深度: {current_depth:.1f} m")

        # 更新波束显示
        self.beam_curve.setData(self.beam_angles, self.beam_data)

        # 更新波束置信度 (模拟一些低置信度点)
        confidence_x = []
        confidence_y = []
        for i in range(len(self.beam_angles)):
            if np.random.rand() > 0.8:  # 随机选择20%的点显示置信度问题
                confidence_x.append(self.beam_angles[i])
                confidence_y.append(self.beam_data[i])
        self.beam_confidence.setData(confidence_x, confidence_y)

        # 更新航迹显示
        self.track_curve.setData(self.track_x, self.track_y)

        # 更新当前位置指示
        if self.track_x:
            self.current_pos_indicator.setData([self.track_x[-1]], [self.track_y[-1]],
                                               symbol='o', symbolSize=10, symbolBrush='r')

        # 更新水深图
        x_idx = min(int(self.position_x * 5) % 50, 49)
        y_idx = min(int(self.position_y * 5) % 50, 49)
        # 在当前位置附近添加新的水深数据
        radius = 3
        for i in range(max(0, x_idx - radius), min(50, x_idx + radius)):
            for j in range(max(0, y_idx - radius), min(50, y_idx + radius)):
                if np.sqrt((i - x_idx) ** 2 + (j - y_idx) ** 2) <= radius:
                    # 模拟测得新的水深数据
                    self.depth_data[j, i] = self.beam_data[len(self.beam_data) // 2] + np.random.rand() * 2 - 1

        self.depth_image.setImage(self.depth_data, autoLevels=False, levels=(10, 30))

        # 更新深度范围标签
        min_depth = np.min(self.depth_data)
        max_depth = np.max(self.depth_data)
        self.depth_range_label.setText(f"深度范围: {min_depth:.1f} - {max_depth:.1f} m")

    def update_device_status(self):
        """更新设备状态显示"""
        # 更新设备状态表格
        for device, status in self.device_status.items():
            item = self.status_cells[device]
            item.setText(status)
            if status == "正常":
                item.setForeground(QColor("green"))
            else:
                item.setForeground(QColor("red"))

        # 检查整体系统状态
        if "警告" in self.device_status.values():
            self.system_status_indicator.setText("系统状态: 警告")
            self.system_status_indicator.setStyleSheet("color: yellow; font-weight: bold; font-size: 14pt")
        else:
            self.system_status_indicator.setText("系统状态: 正常")
            self.system_status_indicator.setStyleSheet("color: lime; font-weight: bold; font-size: 14pt")

        # 模拟CPU和内存使用
        cpu = 30 + int(np.random.rand() * 20)
        self.cpu_usage.setValue(cpu)
        if cpu > 80:
            self.cpu_usage.setStyleSheet("QProgressBar::chunk { background-color: red; }")
        elif cpu > 60:
            self.cpu_usage.setStyleSheet("QProgressBar::chunk { background-color: yellow; }")
        else:
            self.cpu_usage.setStyleSheet("QProgressBar::chunk { background-color: green; }")

        mem = 25 + int(np.random.rand() * 15)
        self.memory_usage.setValue(mem)

        # 更新数据传输率
        data_rate = 2.0 + np.random.rand() * 1.0
        self.data_rate.setText(f"数据传输率: {data_rate:.1f} MB/s")

    def update_3d_view(self):
        """更新3D视图"""
        self.ax3d.clear()

        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.depth_data

        # 获取纵向夸张系数
        exag = self.exaggeration_slider.value()

        # 根据显示选项决定如何绘制
        if self.show_wireframe.isChecked():
            self.surf = self.ax3d.plot_wireframe(X, Y, Z * exag, color='cyan', linewidth=0.5)
        else:
            self.surf = self.ax3d.plot_surface(X, Y, Z * exag, cmap='viridis',
                                               edgecolor='none', alpha=0.7)

        # 显示等高线
        if self.show_contours.isChecked():
            contour_offset = np.max(Z * exag) + 2  # 在表面上方显示等高线
            contours = self.ax3d.contour(X, Y, Z * exag,
                                         zdir='z', offset=contour_offset,
                                         cmap='plasma', linewidths=2)

        # 显示剖面
        if self.show_profile.isChecked():
            profile_idx = self.profile_slider.value()
            profile_x = x
            profile_y = np.ones_like(x) * y[profile_idx]
            profile_z = Z[profile_idx, :] * exag
            self.ax3d.plot(profile_x, profile_y, profile_z, 'r-', linewidth=3)

        # 设置轴标签和标题
        self.ax3d.set_xlabel('X (m)', color='white')
        self.ax3d.set_ylabel('Y (m)', color='white')
        self.ax3d.set_zlabel(f'Depth (m) [Exag:{exag}]', color='white')  # 使用英文
        self.ax3d.set_title('3D Seabed Topography', color='white')  # 使用英文
        self.ax3d.tick_params(colors='white')

        # 设置Z轴方向
        self.ax3d.invert_zaxis()

        # 添加航迹标记
        if len(self.track_x) > 0:
            # 归一化轨迹点到绘图范围
            norm_x = np.interp(self.track_x, [min(self.track_x), max(self.track_x)], [0, 10])
            norm_y = np.interp(self.track_y, [min(self.track_y), max(self.track_y)], [0, 10])

            # 获取轨迹点对应的深度
            track_z = []
            for nx, ny in zip(norm_x, norm_y):
                ix = int(nx / 10 * 49)
                iy = int(ny / 10 * 49)
                ix = max(0, min(ix, 49))
                iy = max(0, min(iy, 49))
                track_z.append(Z[iy, ix] * exag)

            self.ax3d.plot(norm_x, norm_y, track_z, 'r-', linewidth=2)

            # 添加当前位置标记
            if len(norm_x) > 0:
                self.ax3d.scatter([norm_x[-1]], [norm_y[-1]], [track_z[-1]],
                                  color='yellow', s=100, marker='o')

        self.canvas.draw()

    def toggle_acquisition(self):
        """切换数据采集状态"""
        self.acquisition_active = not self.acquisition_active
        if self.acquisition_active:
            self.start_stop_action.setText("暂停采集")
            self.statusBar.showMessage("数据采集已启动")
            self.add_log("数据采集已启动")
        else:
            self.start_stop_action.setText("开始采集")
            self.statusBar.showMessage("数据采集已暂停")
            self.add_log("数据采集已暂停")

    def update_clock(self):
        """更新时钟显示"""
        # 显示当前时间
        current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.clock_label.setText(current_time)

        # 更新测量时长
        elapsed = self.start_time.secsTo(QDateTime.currentDateTime())
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.survey_time_label.setText(f"测量时长: {duration}")
        self.stats["survey_duration"] = duration

    def update_colormap(self, cmap_name):
        """更新水深图的颜色映射"""
        self.depth_image.setColorMap(pg.colormap.get(cmap_name))
        self.statusBar.showMessage(f"已切换颜色映射: {cmap_name}")

    def update_contrast(self):
        """更新水深图的对比度"""
        contrast = self.contrast_slider.value() / 50.0
        depth_min = np.min(self.depth_data)
        depth_max = np.max(self.depth_data)
        depth_range = depth_max - depth_min

        # 调整深度范围以改变对比度
        new_min = depth_min + depth_range * (1 - contrast) * 0.5
        new_max = depth_max - depth_range * (1 - contrast) * 0.5

        self.depth_image.setLevels(min=new_min, max=new_max)

    def toggle_profile_view(self):
        """切换是否显示纵向剖面"""
        show_profile = self.show_profile.isChecked()
        self.profile_slider.setEnabled(show_profile)
        self.update_3d_view()

    def add_log(self, message, log_type="信息"):
        """添加系统日志"""
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.system_log.append((timestamp, log_type, message))

        # 更新日志表格
        row = None
        # self.log_table.insertRow(row)
        # self.log_table.setItem(row, 0, QTableWidgetItem(timestamp))

        type_item = QTableWidgetItem(log_type)
        if log_type == "警告" or log_type == "错误":
            type_item.setForeground(QColor("red"))
        elif log_type == "成功":
            type_item.setForeground(QColor("green"))
        # self.log_table.setItem(row, 1, type_item)

        # self.log_table.setItem(row, 2, QTableWidgetItem(message))

        # 自动滚动到最新的日志
        # self.log_table.scrollToBottom()

    def update_stats_display(self):
        """更新统计信息显示"""
        for key, label in self.stats_display.items():
            if key == "points_collected":
                label.setText(f"{self.stats[key]} 个")
            elif key == "survey_duration":
                label.setText(f"{self.stats[key]}")
            elif key == "coverage_area":
                label.setText(f"{self.stats[key]:.2f} m²")
            else:
                label.setText(f"{self.stats[key]:.2f} m")

    def quick_save(self):
        """快速保存当前数据"""
        # 使用当前时间作为默认文件名
        default_filename = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss") + ".csv"
        self.save_data(default_filename)
        self.statusBar.showMessage(f"数据已快速保存: {default_filename}")

    def save_data(self, default_name=None):
        """保存当前数据"""
        if default_name:
            filename = os.path.join("data", default_name)
            # 确保目录存在
            os.makedirs("data", exist_ok=True)
        else:
            filename, _ = QFileDialog.getSaveFileName(self, "保存数据", "", "CSV Files (*.csv);;All Files (*)")

        if filename:
            try:
                # 创建包含当前数据的DataFrame
                data = {
                    'track_x': self.track_x,
                    'track_y': self.track_y,
                    'beam_data': [list(self.beam_data) for _ in range(len(self.track_x))],  # 每个点保存当前深度数据
                    'timestamp': [time.time() + i for i in range(len(self.track_x))],
                    'gps_lat': [self.gps_lat + i * 0.0001 for i in range(len(self.track_x))],
                    'gps_lon': [self.gps_lon + i * 0.0001 for i in range(len(self.track_x))]
                }

                # 对齐数据长度
                min_len = min(len(self.track_x), len(self.track_y))
                df = pd.DataFrame({
                    'track_x': self.track_x[:min_len],
                    'track_y': self.track_y[:min_len],
                    'gps_lat': data['gps_lat'][:min_len],
                    'gps_lon': data['gps_lon'][:min_len],
                    'timestamp': data['timestamp'][:min_len]
                })

                # 保存
                df.to_csv(filename, index=False)

                # 同时保存水深图
                depth_filename = filename.replace('.csv', '_depth.npy')
                np.save(depth_filename, self.depth_data)

                self.add_log(f"数据已保存到 {filename} 和 {depth_filename}", "成功")
                self.refresh_history()

                return True
            except Exception as e:
                self.add_log(f"保存数据失败: {str(e)}", "错误")
                QMessageBox.critical(self, "保存失败", f"无法保存数据: {str(e)}")
                return False
        return False

    def load_data(self):
        """加载历史数据"""
        filename, _ = QFileDialog.getOpenFileName(self, "加载数据", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            try:
                # 加载航迹数据
                df = pd.read_csv(filename)
                self.track_x = df['track_x'].tolist()
                self.track_y = df['track_y'].tolist()

                # 如果有GPS数据就加载
                if 'gps_lat' in df.columns and 'gps_lon' in df.columns:
                    self.gps_lat = df['gps_lat'].iloc[-1]
                    self.gps_lon = df['gps_lon'].iloc[-1]

                # 尝试加载水深图
                depth_filename = filename.replace('.csv', '_depth.npy')
                if os.path.exists(depth_filename):
                    self.depth_data = np.load(depth_filename)

                # 更新显示
                self.update_realtime_display()
                self.update_3d_view()

                # 更新统计数据
                self.stats["points_collected"] = len(self.track_x)
                if len(self.track_x) >= 2:
                    track_length = np.sum(np.sqrt(np.diff(self.track_x) ** 2 + np.diff(self.track_y) ** 2))
                    self.stats["coverage_area"] = track_length * 2 * 75 * np.pi / 180

                self.add_log(f"已加载数据，共 {len(self.track_x)} 个点", "成功")
                self.statusBar.showMessage(f"已加载数据 {filename}")

            except Exception as e:
                self.add_log(f"加载数据失败: {str(e)}", "错误")
                QMessageBox.critical(self, "加载失败", f"无法加载数据: {str(e)}")

    def export_3d_model(self):
        """导出3D模型"""
        filename, _ = QFileDialog.getSaveFileName(self, "导出3D模型", "", "OBJ Files (*.obj);;All Files (*)")
        if filename:
            try:
                # 此处应实现为真实的3D模型导出
                # 这里只做简单的模拟
                with open(filename, 'w') as f:
                    f.write("# 3D model export from MultibeamSonarSystem\n")

                    # 写入顶点
                    x = np.linspace(0, 10, 50)
                    y = np.linspace(0, 10, 50)
                    X, Y = np.meshgrid(x, y)
                    Z = self.depth_data

                    for i in range(50):
                        for j in range(50):
                            f.write(f"v {X[i, j]} {Y[i, j]} {Z[i, j]}\n")

                    # 写入面
                    for i in range(49):
                        for j in range(49):
                            idx1 = i * 50 + j + 1
                            idx2 = i * 50 + j + 2
                            idx3 = (i + 1) * 50 + j + 2
                            idx4 = (i + 1) * 50 + j + 1
                            f.write(f"f {idx1} {idx2} {idx3} {idx4}\n")

                self.add_log(f"3D模型已导出到 {filename}", "成功")
                QMessageBox.information(self, "导出成功", f"3D模型已成功导出到：\n{filename}")

            except Exception as e:
                self.add_log(f"导出3D模型失败: {str(e)}", "错误")
                QMessageBox.critical(self, "导出失败", f"无法导出3D模型: {str(e)}")

    def refresh_history(self):
        """刷新历史数据列表"""
        # 清空表格
        self.history_table.setRowCount(0)

        # 检查数据目录
        if not os.path.exists("data"):
            os.makedirs("data")
            return

        # 查找所有CSV文件
        csv_files = [f for f in os.listdir("data") if f.endswith(".csv")]

        for csv_file in csv_files:
            try:
                filepath = os.path.join("data", csv_file)
                df = pd.read_csv(filepath)

                row = self.history_table.rowCount()
                self.history_table.insertRow(row)

                # 文件名
                self.history_table.setItem(row, 0, QTableWidgetItem(csv_file))

                # 记录时间（从文件名中提取或使用修改时间）
                if "_" in csv_file:
                    time_str = csv_file.split("_")[0]
                    self.history_table.setItem(row, 1, QTableWidgetItem(time_str))
                else:
                    mtime = os.path.getmtime(filepath)
                    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
                    self.history_table.setItem(row, 1, QTableWidgetItem(time_str))

                # 数据点数
                self.history_table.setItem(row, 2, QTableWidgetItem(str(len(df))))

                # 暂时跳过水深信息，实际应用中应从数据中提取
                self.history_table.setItem(row, 3, QTableWidgetItem("N/A"))
                self.history_table.setItem(row, 4, QTableWidgetItem("N/A"))

                # 添加加载按钮
                load_btn = QPushButton("加载")
                load_btn.clicked.connect(lambda checked, f=filepath: self.load_specific_file(f))
                self.history_table.setCellWidget(row, 5, load_btn)

            except Exception as e:
                print(f"处理文件 {csv_file} 时发生错误: {str(e)}")

        self.add_log(f"已刷新历史数据列表，共 {self.history_table.rowCount()} 条记录")

    def load_specific_file(self, filepath):
        """加载特定文件"""
        try:
            # 加载航迹数据
            df = pd.read_csv(filepath)
            self.track_x = df['track_x'].tolist()
            self.track_y = df['track_y'].tolist()

            # 如果有GPS数据就加载
            if 'gps_lat' in df.columns and 'gps_lon' in df.columns:
                self.gps_lat = df['gps_lat'].iloc[-1]
                self.gps_lon = df['gps_lon'].iloc[-1]

            # 尝试加载水深图
            depth_filename = filepath.replace('.csv', '_depth.npy')
            if os.path.exists(depth_filename):
                self.depth_data = np.load(depth_filename)

            # 更新显示
            self.update_realtime_display()
            self.update_3d_view()

            # 更新统计数据
            self.stats["points_collected"] = len(self.track_x)

            self.add_log(f"已加载数据，共 {len(self.track_x)} 个点", "成功")
            self.statusBar.showMessage(f"已加载数据 {filepath}")

        except Exception as e:
            self.add_log(f"加载数据失败: {str(e)}", "错误")
            QMessageBox.critical(self, "加载失败", f"无法加载数据: {str(e)}")

    def switch_analysis(self, index):
        """切换分析类型"""
        analysis_type = self.analysis_type.currentText()

        # 清除当前图表
        for ax in self.analysis_figure.get_axes():
            ax.clear()

        if analysis_type == "深度趋势分析":
            # 重新创建深度趋势图表
            self.depth_trend_ax = self.analysis_figure.add_subplot(111)
            self.depth_trend_ax.set_facecolor('#2D2D2D')
            self.depth_trend_ax.tick_params(colors='white')
            self.depth_trend_ax.set_xlabel('Time', color='white')  # 使用英文
            self.depth_trend_ax.set_ylabel('Depth (m)', color='white')  # 使用英文
            self.depth_trend_ax.set_title('Depth Trend Analysis', color='white')  # 使用英文

        elif analysis_type == "地形坡度分析":
            # 创建坡度分析图表
            self.depth_trend_ax = self.analysis_figure.add_subplot(111)
            self.depth_trend_ax.set_facecolor('#2D2D2D')
            self.depth_trend_ax.tick_params(colors='white')
            self.depth_trend_ax.set_xlabel('X (m)', color='white')
            self.depth_trend_ax.set_ylabel('Slope (degrees)', color='white')  # 使用英文
            self.depth_trend_ax.set_title('Terrain Slope Analysis', color='white')  # 使用英文

        elif analysis_type == "声速剖面分析":
            # 创建声速分析图表
            self.depth_trend_ax = self.analysis_figure.add_subplot(111)
            self.depth_trend_ax.set_facecolor('#2D2D2D')
            self.depth_trend_ax.tick_params(colors='white')
            self.depth_trend_ax.set_xlabel('Sound Velocity (m/s)', color='white')  # 使用英文
            self.depth_trend_ax.set_ylabel('Depth (m)', color='white')  # 使用英文
            self.depth_trend_ax.set_title('Sound Velocity Profile Analysis', color='white')  # 使用英文
            self.depth_trend_ax.invert_yaxis()  # 反转Y轴，深度增加向下

        elif analysis_type == "测量精度分析":
            # 创建测量精度分析图表
            self.depth_trend_ax = self.analysis_figure.add_subplot(111)
            self.depth_trend_ax.set_facecolor('#2D2D2D')
            self.depth_trend_ax.tick_params(colors='white')
            self.depth_trend_ax.set_xlabel('Time', color='white')  # 使用英文
            self.depth_trend_ax.set_ylabel('Accuracy (m)', color='white')  # 使用英文
            self.depth_trend_ax.set_title('Measurement Accuracy Analysis', color='white')  # 使用英文

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def run_analysis(self):
        """执行数据分析"""
        analysis_type = self.analysis_type.currentText()

        # 清除当前图表
        self.depth_trend_ax.clear()

        if analysis_type == "深度趋势分析":
            # 分析深度趋势
            if len(self.history_depth) > 0:
                x = np.arange(len(self.history_depth))
                y = self.history_depth

                # 绘制深度值
                self.depth_trend_ax.plot(x, y, 'b-', label='Raw Depth')  # 使用英文

                # 计算移动平均
                if len(y) >= 10:
                    window = 10
                    y_smooth = np.convolve(y, np.ones(window) / window, mode='valid')
                    x_smooth = x[window - 1:]
                    self.depth_trend_ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Moving Average')  # 使用英文

                # 计算趋势线
                if len(y) >= 2:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    self.depth_trend_ax.plot(x, p(x), "g--", linewidth=2,
                                             label=f'Trend {z[0]:.4f}x + {z[1]:.2f}')  # 使用英文

                self.depth_trend_ax.legend()
                self.depth_trend_ax.set_xlabel('Time', color='white')  # 使用英文
                self.depth_trend_ax.set_ylabel('Depth (m)', color='white')  # 使用英文
                self.depth_trend_ax.set_title('Depth Trend Analysis', color='white')  # 使用英文

                # 更新统计信息
                avg_depth = np.mean(y)
                std_depth = np.std(y)
                max_depth = np.max(y)
                min_depth = np.min(y)

                self.stats_labels["平均深度"].setText(f"{avg_depth:.2f} m")
                self.stats_labels["最大深度"].setText(f"{max_depth:.2f} m")
                self.stats_labels["最小深度"].setText(f"{min_depth:.2f} m")
                self.stats_labels["深度标准差"].setText(f"{std_depth:.2f} m")

        elif analysis_type == "地形坡度分析":
            # 计算地形坡度
            gradient_x = np.gradient(self.depth_data, axis=1)
            gradient_y = np.gradient(self.depth_data, axis=0)
            slope = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
            slope_degrees = np.degrees(np.arctan(slope))

            # 绘制坡度图
            im = self.depth_trend_ax.imshow(slope_degrees, cmap='hot',
                                            interpolation='nearest', aspect='auto')
            cbar = plt.colorbar(im, ax=self.depth_trend_ax)
            cbar.set_label('Slope (degrees)', color='white')  # 使用英文
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')

            self.depth_trend_ax.set_xlabel('X', color='white')
            self.depth_trend_ax.set_ylabel('Y', color='white')
            self.depth_trend_ax.set_title('Terrain Slope Analysis', color='white')  # 使用英文

            # 更新统计信息
            avg_slope = np.mean(slope_degrees)
            max_slope = np.max(slope_degrees)
            min_slope = np.min(slope_degrees)
            std_slope = np.std(slope_degrees)

            self.stats_labels["坡度平均值"].setText(f"{avg_slope:.2f}°")
            self.stats_labels["最大坡度"].setText(f"{max_slope:.2f}°")
            self.stats_labels["最小坡度"].setText(f"{min_slope:.2f}°")
            self.stats_labels["坡度标准差"].setText(f"{std_slope:.2f}°")

        elif analysis_type == "声速剖面分析":
            # 绘制声速剖面
            depth = self.sound_velocity_profile["深度"]
            velocity = self.sound_velocity_profile["声速"]

            self.depth_trend_ax.plot(velocity, depth, 'r-', linewidth=2)
            self.depth_trend_ax.set_xlabel('Sound Velocity (m/s)', color='white')  # 使用英文
            self.depth_trend_ax.set_ylabel('Depth (m)', color='white')  # 使用英文
            self.depth_trend_ax.set_title('Sound Velocity Profile Analysis', color='white')  # 使用英文
            self.depth_trend_ax.grid(True, linestyle='--', alpha=0.7)
            self.depth_trend_ax.invert_yaxis()  # 反转Y轴，深度增加向下

        elif analysis_type == "测量精度分析":
            # 模拟测量精度数据
            if len(self.history_depth) > 0:
                x = np.arange(len(self.history_depth))

                # 模拟随时间变化的精度
                accuracy = 0.05 + 0.02 * np.random.rand(len(x))

                # 添加一些随机的精度变化
                for i in range(5):
                    pos = np.random.randint(0, len(x))
                    width = np.random.randint(5, 20)
                    for j in range(max(0, pos - width // 2), min(len(x), pos + width // 2)):
                        if j < len(accuracy):
                            accuracy[j] += 0.1 * np.random.rand()

                self.depth_trend_ax.plot(x, accuracy, 'g-', linewidth=2)

                # 添加精度阈值线
                self.depth_trend_ax.axhline(y=0.1, color='r', linestyle='--', label='Threshold')  # 使用英文

                self.depth_trend_ax.fill_between(x, 0, accuracy, alpha=0.3, color='green')
                self.depth_trend_ax.set_xlabel('Time', color='white')  # 使用英文
                self.depth_trend_ax.set_ylabel('Accuracy (m)', color='white')  # 使用英文
                self.depth_trend_ax.set_title('Measurement Accuracy Analysis', color='white')  # 使用英文
                self.depth_trend_ax.legend()

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

        self.add_log(f"已执行 {analysis_type}")

    def update_system_param(self, key, value):
        """更新系统参数"""
        self.system_params[key] = value
        # 更新显示
        slider, label = self.param_controls[key]
        label.setText(f"{value}")

        # 记录更改
        self.add_log(f"系统参数已更改: {key} = {value}")

        # 如果是声速参数，更新声速剖面
        if key == "声速":
            # 更新基准声速，保持剖面形状
            base_velocity = self.sound_velocity_profile["声速"].mean()
            delta = value - base_velocity
            self.sound_velocity_profile["声速"] += delta
            # 更新声速剖面图
            self.svp_line.set_xdata(self.sound_velocity_profile["声速"])
            self.svp_canvas.draw()

    def setup_matplotlib_chinese_support(self):
        """尝试设置matplotlib支持中文"""
        try:
            # 尝试不同的中文字体
            font_options = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']

            for font in font_options:
                try:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                    # 测试字体是否工作
                    fig = Figure()
                    ax = fig.add_subplot(111)
                    ax.set_title('测试')
                    fig.canvas.draw()
                    # 如果没有抛出异常，则字体可用
                    print(f"成功配置matplotlib支持中文，使用字体: {font}")
                    return True
                except:
                    continue

            # 如果所有字体都失败了
            print("无法找到支持中文的字体，将使用英文标签")
            return False
        except Exception as e:
            print(f"配置matplotlib中文支持时出错: {str(e)}")
            return False

    def load_svp(self):
        """加载声速剖面数据"""
        filename, _ = QFileDialog.getOpenFileName(self, "加载声速剖面", "", "CSV Files (*.csv);;All Files (*)")
        if filename:
            try:
                # 加载声速剖面数据
                df = pd.read_csv(filename)

                if 'depth' in df.columns and 'velocity' in df.columns:
                    self.sound_velocity_profile["深度"] = df['depth'].values
                    self.sound_velocity_profile["声速"] = df['velocity'].values

                    # 更新声速剖面图
                    self.svp_ax.clear()
                    self.svp_ax.plot(self.sound_velocity_profile["声速"],
                                     self.sound_velocity_profile["深度"], 'r-')
                    self.svp_ax.set_xlabel('声速 (m/s)', color='white')
                    self.svp_ax.set_ylabel('深度 (m)', color='white')
                    self.svp_ax.set_title('声速剖面', color='white')
                    self.svp_ax.invert_yaxis()
                    self.svp_canvas.draw()

                    # 更新系统参数中的平均声速
                    self.system_params["声速"] = np.mean(self.sound_velocity_profile["声速"])
                    slider, label = self.param_controls["声速"]
                    slider.setValue(int(self.system_params["声速"]))

                    self.add_log(f"已加载声速剖面: {filename}", "成功")

                else:
                    raise ValueError("文件格式不正确，需要包含'depth'和'velocity'列")

            except Exception as e:
                self.add_log(f"加载声速剖面失败: {str(e)}", "错误")
                QMessageBox.critical(self, "加载失败", f"无法加载声速剖面: {str(e)}")

    def edit_svp(self):
        """编辑声速剖面数据"""
        # 创建编辑对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("编辑声速剖面")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 添加说明
        layout.addWidget(QLabel("编辑声速剖面数据（格式：'深度,声速' 每行一对）"))

        # 添加文本编辑框
        text_edit = QTextEdit()
        # 填充现有数据
        text_content = ""
        for d, v in zip(self.sound_velocity_profile["深度"], self.sound_velocity_profile["声速"]):
            text_content += f"{d},{v}\n"
        text_edit.setText(text_content)

        layout.addWidget(text_edit)

        # 添加按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # 显示对话框并处理结果
        if dialog.exec_() == QDialog.Accepted:
            try:
                # 解析编辑后的数据
                content = text_edit.toPlainText().strip().split('\n')
                depths = []
                velocities = []

                for line in content:
                    if ',' in line:
                        d, v = line.split(',')
                        depths.append(float(d))
                        velocities.append(float(v))

                if len(depths) > 1:  # 至少需要两个点
                    self.sound_velocity_profile["深度"] = np.array(depths)
                    self.sound_velocity_profile["声速"] = np.array(velocities)

                    # 更新声速剖面图
                    self.svp_ax.clear()
                    self.svp_line, = self.svp_ax.plot(self.sound_velocity_profile["声速"],
                                                      self.sound_velocity_profile["深度"], 'r-')
                    self.svp_ax.set_xlabel('声速 (m/s)', color='white')
                    self.svp_ax.set_ylabel('深度 (m)', color='white')
                    self.svp_ax.set_title('声速剖面', color='white')
                    self.svp_ax.invert_yaxis()
                    self.svp_canvas.draw()

                    # 更新系统参数中的平均声速
                    self.system_params["声速"] = np.mean(self.sound_velocity_profile["声速"])
                    slider, label = self.param_controls["声速"]
                    slider.setValue(int(self.system_params["声速"]))

                    self.add_log("声速剖面已更新", "成功")
                else:
                    raise ValueError("至少需要两个数据点")

            except Exception as e:
                self.add_log(f"更新声速剖面失败: {str(e)}", "错误")
                QMessageBox.critical(self, "更新失败", f"无法更新声速剖面: {str(e)}")

    def calibrate_system(self):
        """系统校准"""
        dialog = QDialog(self)
        dialog.setWindowTitle("系统校准")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 添加校准选项
        layout.addWidget(QLabel("选择校准类型:"))

        calib_type = QComboBox()
        calib_type.addItems(["声速校准", "姿态校准", "位置校准", "全面校准"])
        layout.addWidget(calib_type)

        # 添加校准标准区域选择
        layout.addWidget(QLabel("选择校准标准区域:"))

        standard_area = QComboBox()
        standard_area.addItems(["标准测试区A", "标准测试区B", "用户自定义..."])
        layout.addWidget(standard_area)

        # 添加按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # 显示对话框并处理结果
        if dialog.exec_() == QDialog.Accepted:
            # 模拟校准过程
            self.statusBar.showMessage("系统校准中...")

            # 创建并显示进度对话框
            progress = QDialog(self)
            progress.setWindowTitle("校准进行中")
            progress_layout = QVBoxLayout()

            progress_label = QLabel("正在进行校准，请稍候...")
            progress_layout.addWidget(progress_label)

            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            progress_layout.addWidget(progress_bar)

            progress.setLayout(progress_layout)
            progress.show()

            # 模拟校准过程
            for i in range(101):
                progress_bar.setValue(i)
                if i % 10 == 0:
                    progress_label.setText(f"校准进度: {i}%")
                QApplication.processEvents()
                time.sleep(0.05)

            progress.close()

            # 校准完成后更新状态
            self.add_log(f"系统校准完成: {calib_type.currentText()}", "成功")
            QMessageBox.information(self, "校准完成", f"{calib_type.currentText()} 已成功完成!")
            self.statusBar.showMessage("系统校准完成")

    def filter_data(self):
        """数据过滤处理"""
        dialog = QDialog(self)
        dialog.setWindowTitle("数据过滤")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 添加过滤选项
        layout.addWidget(QLabel("选择过滤类型:"))

        filter_type = QComboBox()
        filter_type.addItems(["噪声过滤", "离群值移除", "平滑处理", "综合过滤"])
        layout.addWidget(filter_type)

        # 添加过滤强度选择
        layout.addWidget(QLabel("过滤强度:"))

        filter_strength = QSlider(Qt.Horizontal)
        filter_strength.setRange(1, 10)
        filter_strength.setValue(5)
        filter_strength.setTickPosition(QSlider.TicksBelow)
        filter_strength.setTickInterval(1)
        layout.addWidget(filter_strength)

        strength_label = QLabel("中等 (5)")
        layout.addWidget(strength_label)

        # 更新强度标签
        filter_strength.valueChanged.connect(
            lambda v: strength_label.setText({
                                                 1: "最弱 (1)",
                                                 2: "很弱 (2)",
                                                 3: "弱 (3)",
                                                 4: "中偏弱 (4)",
                                                 5: "中等 (5)",
                                                 6: "中偏强 (6)",
                                                 7: "强 (7)",
                                                 8: "很强 (8)",
                                                 9: "极强 (9)",
                                                 10: "最强 (10)"
                                             }[v])
        )

        # 添加按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # 显示对话框并处理结果
        if dialog.exec_() == QDialog.Accepted:
            filter_name = filter_type.currentText()
            strength = filter_strength.value()

            # 应用过滤
            if filter_name == "噪声过滤":
                # 应用高斯过滤
                sigma = 0.5 * strength
                self.depth_data = signal.gaussian_filter(self.depth_data, sigma)

            elif filter_name == "离群值移除":
                # 移除离群值
                threshold = 0.2 * strength
                mean = np.mean(self.depth_data)
                std = np.std(self.depth_data)
                mask = np.abs(self.depth_data - mean) > threshold * std
                self.depth_data[mask] = np.median(self.depth_data)

            elif filter_name == "平滑处理":
                # 应用中值过滤
                kernel_size = 1 + 2 * strength
                self.depth_data = signal.medfilt2d(self.depth_data, kernel_size=kernel_size)

            elif filter_name == "综合过滤":
                # 同时应用多种过滤
                # 先移除离群值
                threshold = 0.2 * strength
                mean = np.mean(self.depth_data)
                std = np.std(self.depth_data)
                mask = np.abs(self.depth_data - mean) > threshold * std
                self.depth_data[mask] = np.median(self.depth_data)

                # 再应用高斯平滑
                sigma = 0.3 * strength
                self.depth_data = signal.gaussian_filter(self.depth_data, sigma)

            # 更新显示
            self.depth_image.setImage(self.depth_data, autoLevels=False)
            self.update_3d_view()

            self.add_log(f"已应用滤波: {filter_name} (强度 {strength})", "成功")
            self.statusBar.showMessage(f"数据过滤完成: {filter_name}")

    def export_results(self):
        """导出测量结果"""
        dialog = QDialog(self)
        dialog.setWindowTitle("导出测量结果")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout()

        # 添加导出选项
        layout.addWidget(QLabel("选择导出格式:"))

        export_format = QComboBox()
        export_format.addItems(["CSV", "Excel", "PDF报告", "图像"])
        layout.addWidget(export_format)

        # 添加导出内容选择
        layout.addWidget(QLabel("选择导出内容:"))

        content_layout = QVBoxLayout()

        export_track = QCheckBox("航迹数据")
        export_track.setChecked(True)
        content_layout.addWidget(export_track)

        export_depth = QCheckBox("水深数据")
        export_depth.setChecked(True)
        content_layout.addWidget(export_depth)

        export_3d = QCheckBox("3D模型")
        content_layout.addWidget(export_3d)

        export_stats = QCheckBox("统计数据")
        export_stats.setChecked(True)
        content_layout.addWidget(export_stats)

        layout.addLayout(content_layout)

        # 添加按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        layout.addWidget(buttons)
        dialog.setLayout(layout)

        # 显示对话框并处理结果
        if dialog.exec_() == QDialog.Accepted:
            format_name = export_format.currentText()

            # 根据选择的格式决定文件扩展名
            if format_name == "CSV":
                ext = "csv"
            elif format_name == "Excel":
                ext = "xlsx"
            elif format_name == "PDF报告":
                ext = "pdf"
            elif format_name == "图像":
                ext = "png"

            filename, _ = QFileDialog.getSaveFileName(self, "导出结果", f"survey_result.{ext}",
                                                      f"{format_name} Files (*.{ext});;All Files (*)")

            if filename:
                try:
                    # 导出数据
                    if format_name == "CSV":
                        # 创建DataFrame存储数据
                        export_data = {}

                        if export_track.isChecked() and self.track_x:
                            export_data['track_x'] = self.track_x
                            export_data['track_y'] = self.track_y

                        if export_depth.isChecked():
                            # 将2D深度数据转为1D数组
                            export_data['depth'] = self.depth_data.flatten()

                        if export_stats.isChecked():
                            for key, value in self.stats.items():
                                if isinstance(value, (int, float)):
                                    export_data[key] = [value] * len(self.track_x)

                        # 创建DataFrame
                        df = pd.DataFrame(export_data)

                        # 保存CSV
                        df.to_csv(filename, index=False)

                    elif format_name == "Excel":
                        # 使用pandas导出到Excel
                        with pd.ExcelWriter(filename) as writer:
                            # 导出航迹数据
                            if export_track.isChecked() and self.track_x:
                                track_df = pd.DataFrame({
                                    'track_x': self.track_x,
                                    'track_y': self.track_y
                                })
                                track_df.to_excel(writer, sheet_name='航迹数据', index=False)

                            # 导出深度数据
                            if export_depth.isChecked():
                                depth_df = pd.DataFrame(self.depth_data)
                                depth_df.to_excel(writer, sheet_name='水深数据')

                            # 导出统计数据
                            if export_stats.isChecked():
                                stats_df = pd.DataFrame([self.stats])
                                stats_df.to_excel(writer, sheet_name='统计数据', index=False)

                    elif format_name == "PDF报告":
                        # 这里简单模拟PDF生成过程
                        with open(filename, 'wb') as f:
                            f.write(b'%PDF-1.4\n%Mock PDF for survey results\n')

                    elif format_name == "图像":
                        # 导出当前3D视图为图像
                        if export_3d.isChecked():
                            self.figure.savefig(filename, dpi=300, facecolor='#2D2D2D')

                    self.add_log(f"测量结果已导出: {filename}", "成功")
                    self.statusBar.showMessage(f"导出完成: {filename}")
                    QMessageBox.information(self, "导出成功", f"结果已成功导出至：\n{filename}")

                except Exception as e:
                    self.add_log(f"导出结果失败: {str(e)}", "错误")
                    QMessageBox.critical(self, "导出失败", f"无法导出结果: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置应用程序图标和主题（如果有图标文件）
    # app.setWindowIcon(QIcon('icon.png'))

    # 创建并显示主窗口
    window = MultibeamSonarSystem()
    window.show()

    # 如果提供了命令行参数，尝试加载指定文件
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        window.load_specific_file(sys.argv[1])

    sys.exit(app.exec_())

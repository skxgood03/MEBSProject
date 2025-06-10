import sys
import numpy as np
import time
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget, QPushButton, QLabel,
                             QGridLayout, QFileDialog, QComboBox, QSlider, QGroupBox, QHBoxLayout, QSplitter,
                             QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QProgressBar, QMenu, QAction,
                             QToolBar, QStatusBar, QDialog, QLineEdit, QFormLayout, QDialogButtonBox, QMessageBox)
from PyQt5.QtCore import QTimer, Qt, QDateTime, pyqtSignal
from PyQt5.QtGui import QIcon, QColor, QPalette, QFont
import pyqtgraph as pg
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import scipy.signal as signal


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
        self.create_analysis_tab()  # 数据分析选项卡
        self.create_settings_tab()  # 设置选项卡

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

        # 应用调色板
        self.setPalette(dark_palette)

        # 设置样式表
        self.setStyleSheet("""
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

            QGroupBox {
                font-weight: bold;
                border: 1px solid #808080;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }

            QLabel {
                color: white;
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

        # 添加时钟显示
        self.clock_label = QLabel("00:00:00")
        self.toolbar.addWidget(self.clock_label)

        # 添加测量时间
        self.survey_time_label = QLabel("测量时长: 00:00:00")
        self.toolbar.addWidget(self.survey_time_label)

    def init_data(self):
        """初始化数据"""
        # 采集状态
        self.acquisition_active = True

        # 模拟声呐位置
        self.position_x = 0
        self.position_y = 0
        self.vessel_heading = 0  # 船舶航向

        # GPS数据
        self.gps_lat = 30.0  # 起始纬度
        self.gps_lon = 120.0  # 起始经度

        # 模拟航迹数据
        self.track_x = []
        self.track_y = []

        # 模拟水深数据矩阵 (50x50)
        self.depth_data = np.zeros((50, 50))
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        self.depth_data = 20 + 5 * np.sin(X) + 3 * np.cos(Y) + np.random.rand(50, 50) * 2

        # 模拟数据
        self.beam_count = 64  # 增加波束数量到64
        self.beam_data = np.random.rand(self.beam_count) * 10 + 20
        self.beam_angles = np.linspace(-75, 75, self.beam_count)

        # 系统参数
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
        self.depth_trend_ax.set_xlabel('Time', color='white')
        self.depth_trend_ax.set_ylabel('Depth (m)', color='white')
        self.depth_trend_ax.set_title('Depth Trend Analysis', color='white')
        self.analysis_figure.tight_layout()

        main_layout.addWidget(self.analysis_canvas)

        # 添加分析结果和统计信息显示区域
        results_group = QGroupBox("分析结果")
        results_layout = QGridLayout()

        # 添加统计信息标签
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
            slider.setValue(current_value)
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
        self.svp_ax.set_xlabel('Sound Velocity (m/s)', color='white')
        self.svp_ax.set_ylabel('Depth (m)', color='white')
        self.svp_ax.set_title('Sound Velocity Profile', color='white')
        self.svp_ax.invert_yaxis()

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

    def update_data(self):
        """更新模拟数据"""
        if not self.acquisition_active:
            return

        # 更新GPS位置
        self.gps_lat += np.random.normal(0, 0.0001)
        self.gps_lon += np.random.normal(0, 0.0001)
        self.gps_display.setText(f"GPS: {self.gps_lat:.4f}° N, {self.gps_lon:.4f}° E")

        # 更新航向
        self.vessel_heading += np.random.normal(0, 1)
        self.vessel_heading %= 360
        self.heading_display.setText(f"航向: {self.vessel_heading:.1f}°")

        # 更新多波束数据
        self.beam_data = 20 + 5 * np.sin(self.beam_angles * np.pi / 180) + np.random.rand(self.beam_count)
        self.current_depth_display.setText(f"当前深度: {np.mean(self.beam_data):.1f} m")

        # 更新波束显示
        self.beam_curve.setData(self.beam_angles, self.beam_data)

        # 更新波束置信度显示
        confidence = np.random.rand(self.beam_count) > 0.1
        x_conf = self.beam_angles[confidence]
        y_conf = self.beam_data[confidence]
        self.beam_confidence.setData(x_conf, y_conf)

        # 更新航迹
        self.position_x += np.random.normal(0, 0.1)
        self.position_y += np.random.normal(0, 0.1)
        self.track_x.append(self.position_x)
        self.track_y.append(self.position_y)

        # 限制航迹点数以避免性能问题
        if len(self.track_x) > 1000:
            self.track_x = self.track_x[-1000:]
            self.track_y = self.track_y[-1000:]

        self.track_curve.setData(self.track_x, self.track_y)
        self.current_pos_indicator.setData([self.position_x], [self.position_y])

        # 更新深度图
        self.depth_data = np.roll(self.depth_data, 1, axis=0)
        self.depth_data[0] = 20 + 5 * np.sin(np.linspace(0, 10, 50)) + np.random.rand(50)
        self.depth_image.setImage(self.depth_data, autoLevels=False)

        # 更新统计信息
        self.update_stats()

    def update_stats(self):
        """更新测量统计信息"""
        # 更新点数统计
        self.stats["points_collected"] += self.beam_count

        # 更新深度统计
        current_depths = self.beam_data[~np.isnan(self.beam_data)]
        if len(current_depths) > 0:
            self.stats["max_depth"] = max(self.stats["max_depth"], np.max(current_depths))
            self.stats["min_depth"] = min(self.stats["min_depth"], np.min(current_depths))
            self.stats["avg_depth"] = np.mean(current_depths)

        # 更新测量时间
        elapsed = QDateTime.currentDateTime().secsTo(self.start_time)
        hours = abs(elapsed) // 3600
        minutes = (abs(elapsed) % 3600) // 60
        seconds = abs(elapsed) % 60
        self.stats["survey_duration"] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # 估算覆盖面积
        if len(self.track_x) > 1:
            swath_width = 2 * np.mean(self.beam_data) * np.tan(np.radians(75))  # 75度为最大波束角
            distance = np.sum(np.sqrt(np.diff(self.track_x) ** 2 + np.diff(self.track_y) ** 2))
            self.stats["coverage_area"] = distance * swath_width

    def update_clock(self):
        """更新时钟显示"""
        current_time = QDateTime.currentDateTime()
        self.clock_label.setText(current_time.toString("HH:mm:ss"))

        # 更新测量时长
        elapsed = current_time.secsTo(self.start_time)
        hours = abs(elapsed) // 3600
        minutes = (abs(elapsed) % 3600) // 60
        seconds = abs(elapsed) % 60
        self.survey_time_label.setText(f"测量时长: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def toggle_acquisition(self):
        """切换数据采集状态"""
        self.acquisition_active = not self.acquisition_active
        self.start_stop_action.setText("继续采集" if not self.acquisition_active else "暂停采集")
        status_msg = "数据采集已暂停" if not self.acquisition_active else "数据采集进行中"
        self.statusBar.showMessage(status_msg)

    def update_colormap(self):
        """更新水深图配色方案"""
        colormap = self.color_map_combo.currentText()
        self.depth_image.setColorMap(pg.colormap.get(colormap))

    def update_contrast(self):
        """更新水深图对比度"""
        value = self.contrast_slider.value() / 50.0  # 将滑块值转换为对比度调整因子
        levels = np.percentile(self.depth_data, [2, 98])
        mid = np.mean(levels)
        spread = levels[1] - levels[0]
        new_levels = [mid - spread * value / 2, mid + spread * value / 2]
        self.depth_image.setLevels(new_levels)

    def update_3d_view(self):
        """更新3D视图"""
        self.ax3d.clear()

        # 准备数据
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 10, 50)
        X, Y = np.meshgrid(x, y)
        Z = self.depth_data

        # 设置纵向夸张
        exaggeration = self.exaggeration_slider.value()

        # 绘制表面
        surf = self.ax3d.plot_surface(X, Y, Z * exaggeration, cmap='viridis')

        # 添加等高线
        if self.show_contours.isChecked():
            levels = np.linspace(np.min(Z), np.max(Z), 10)
            self.ax3d.contour(X, Y, Z * exaggeration, levels, colors='w', linestyles='solid', alpha=0.5)

        # 添加网格
        if self.show_wireframe.isChecked():
            self.ax3d.plot_wireframe(X, Y, Z * exaggeration, color='gray', alpha=0.2)

        # 显示剖面
        if self.show_profile.isChecked():
            pos = self.profile_slider.value()
            self.ax3d.plot(x, [y[pos]] * len(x), Z[pos] * exaggeration, 'r-', linewidth=2)

        # 设置标签
        self.ax3d.set_xlabel('X (m)')
        self.ax3d.set_ylabel('Y (m)')
        self.ax3d.set_zlabel('Depth (m)')

        # 刷新画布
        self.canvas.draw()

    def run_analysis(self):
        """执行数据分析"""
        analysis_type = self.analysis_type.currentText()

        # 清空当前图形
        self.depth_trend_ax.clear()

        if analysis_type == "深度趋势分析":
            # 生成示例数据
            times = np.linspace(0, 100, len(self.history_depth))
            depths = np.array(self.history_depth)

            # 绘制深度趋势
            self.depth_trend_ax.plot(times, depths, 'b-', label='实测深度')

            # 添加趋势线
            z = np.polyfit(times, depths, 1)
            p = np.poly1d(z)
            self.depth_trend_ax.plot(times, p(times), 'r--', label='趋势线')

            # 计算统计信息
            self.stats_labels["平均深度"].setText(f"{np.mean(depths):.2f} m")
            self.stats_labels["最大深度"].setText(f"{np.max(depths):.2f} m")
            self.stats_labels["最小深度"].setText(f"{np.min(depths):.2f} m")
            self.stats_labels["深度标准差"].setText(f"{np.std(depths):.2f} m")

        elif analysis_type == "地形坡度分析":
            # 计算坡度
            dy, dx = np.gradient(self.depth_data)
            slope = np.sqrt(dx ** 2 + dy ** 2)

            # 绘制坡度分布图
            img = self.depth_trend_ax.imshow(slope, cmap='viridis')
            self.analysis_figure.colorbar(img, ax=self.depth_trend_ax, label='坡度 (度)')

            # 更新统计信息
            self.stats_labels["坡度平均值"].setText(f"{np.mean(slope):.2f}°")
            self.stats_labels["最大坡度"].setText(f"{np.max(slope):.2f}°")
            self.stats_labels["最小坡度"].setText(f"{np.min(slope):.2f}°")
            self.stats_labels["坡度标准差"].setText(f"{np.std(slope):.2f}°")

        elif analysis_type == "声速剖面分析":
            # 绘制声速剖面
            self.depth_trend_ax.plot(
                self.sound_velocity_profile["声速"],
                self.sound_velocity_profile["深度"],
                'g-'
            )
            self.depth_trend_ax.invert_yaxis()
            self.depth_trend_ax.set_xlabel('声速 (m/s)')
            self.depth_trend_ax.set_ylabel('深度 (m)')

        elif analysis_type == "测量精度分析":
            # 生成示例精度数据
            accuracy = np.random.normal(0.1, 0.02, 100)

            # 绘制直方图
            self.depth_trend_ax.hist(accuracy, bins=20, color='cyan', alpha=0.7)
            self.depth_trend_ax.set_xlabel('测量误差 (m)')
            self.depth_trend_ax.set_ylabel('频次')

            # 添加正态分布拟合
            from scipy import stats
            mu, sigma = stats.norm.fit(accuracy)
            x = np.linspace(min(accuracy), max(accuracy), 100)
            y = stats.norm.pdf(x, mu, sigma) * len(accuracy) * (max(accuracy) - min(accuracy)) / 20
            self.depth_trend_ax.plot(x, y, 'r-', linewidth=2)

        # 更新图形
        self.depth_trend_ax.grid(True)
        self.depth_trend_ax.set_title(analysis_type)
        self.analysis_canvas.draw()

    def quick_save(self):
        """快速保存当前数据"""
        current_time = QDateTime.currentDateTime().toString("yyyyMMdd_hhmmss")
        filename = f"sonar_data_{current_time}.npz"

        try:
            np.savez(filename,
                     depth_data=self.depth_data,
                     beam_data=self.beam_data,
                     track_x=self.track_x,
                     track_y=self.track_y,
                     params=self.system_params,
                     stats=self.stats)
            self.statusBar.showMessage(f"数据已保存至 {filename}")
        except Exception as e:
            QMessageBox.warning(self, "保存失败", f"保存数据时出错：{str(e)}")

    def load_data(self):
        """加载历史数据"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择数据文件", "", "NumPy Files (*.npz)")
        if filename:
            try:
                data = np.load(filename)
                self.depth_data = data['depth_data']
                self.beam_data = data['beam_data']
                self.track_x = data['track_x'].tolist()
                self.track_y = data['track_y'].tolist()
                self.system_params = data['params'].item()
                self.stats = data['stats'].item()

                # 更新显示
                self.depth_image.setImage(self.depth_data)
                self.beam_curve.setData(self.beam_angles, self.beam_data)
                self.track_curve.setData(self.track_x, self.track_y)
                self.update_3d_view()

                self.statusBar.showMessage(f"已加载数据文件 {filename}")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"加载数据时出错：{str(e)}")

    def export_3d_model(self):
        """导出3D模型"""
        filename, _ = QFileDialog.getSaveFileName(self, "导出3D模型", "", "OBJ Files (*.obj)")
        if filename:
            try:
                # 创建OBJ文件
                with open(filename, 'w') as f:
                    # 写入顶点
                    for i in range(50):
                        for j in range(50):
                            f.write(f"v {i / 5} {j / 5} {self.depth_data[i, j]}\n")

                    # 写入面
                    for i in range(49):
                        for j in range(49):
                            v1 = i * 50 + j + 1
                            v2 = i * 50 + j + 2
                            v3 = (i + 1) * 50 + j + 1
                            v4 = (i + 1) * 50 + j + 2
                            f.write(f"f {v1} {v2} {v4}\n")
                            f.write(f"f {v1} {v4} {v3}\n")

                self.statusBar.showMessage(f"3D模型已导出至 {filename}")
            except Exception as e:
                QMessageBox.warning(self, "导出失败", f"导出3D模型时出错：{str(e)}")

    def calibrate_system(self):
        """系统校准"""
        reply = QMessageBox.question(self, "系统校准",
                                     "是否开始系统校准？这将需要几分钟时间。",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 模拟校准过程
            progress = QProgressDialog("正在进行系统校准...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)

            for i in range(101):
                if progress.wasCanceled():
                    break
                progress.setValue(i)
                QApplication.processEvents()
                time.sleep(0.05)

            if not progress.wasCanceled():
                QMessageBox.information(self, "校准完成", "系统校准已完成！")

    def switch_analysis(self):
        """切换分析类型"""
        self.run_analysis()

    def update_system_param(self, key, value):
        """更新系统参数"""
        self.system_params[key] = value
        self.param_controls[key][1].setText(f"{value}")

    def load_svp(self):
        """加载声速剖面"""
        filename, _ = QFileDialog.getOpenFileName(self, "选择声速剖面文件", "", "Text Files (*.txt)")
        if filename:
            try:
                data = np.loadtxt(filename)
                self.sound_velocity_profile["深度"] = data[:, 0]
                self.sound_velocity_profile["声速"] = data[:, 1]
                self.update_svp_plot()
                self.statusBar.showMessage(f"已加载声速剖面 {filename}")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"加载声速剖面时出错：{str(e)}")

    def edit_svp(self):
        """编辑声速剖面"""
        # 这里可以添加声速剖面编辑对话框
        QMessageBox.information(self, "功能未实现", "声速剖面编辑功能正在开发中...")

    def update_svp_plot(self):
        """更新声速剖面图"""
        self.svp_ax.clear()
        self.svp_ax.plot(
            self.sound_velocity_profile["声速"],
            self.sound_velocity_profile["深度"],
            'r-'
        )
        self.svp_ax.invert_yaxis()
        self.svp_ax.set_xlabel('Sound Velocity (m/s)', color='white')
        self.svp_ax.set_ylabel('Depth (m)', color='white')
        self.svp_ax.set_title('Sound Velocity Profile', color='white')
        self.svp_canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置应用程序样式
    app.setStyle("Fusion")

    # 创建主窗口
    window = MultibeamSonarSystem()
    window.show()

    sys.exit(app.exec_())

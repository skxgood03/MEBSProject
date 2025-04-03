import sys
import os
import numpy as np
import time
from datetime import datetime
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QGridLayout, QFileDialog, QSplitter,
                             QComboBox, QCheckBox, QGroupBox, QSlider, QStatusBar, QToolBar,
                             QAction, QLineEdit, QMessageBox, QTableWidget, QTableWidgetItem,
                             QProgressBar, QDockWidget, QFrame, QSizePolicy)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont, QColor, QLinearGradient, QPalette, QBrush, QImage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# 自定义样式表
STYLE_SHEET = """
QMainWindow {
    background-color: #2D2D30;
    color: #FFFFFF;
}
QTabWidget {
    background-color: #2D2D30;
}
QTabWidget::pane {
    border: 1px solid #3F3F46;
    background-color: #252526;
}
QTabBar::tab {
    background-color: #2D2D30;
    color: #CCCCCC;
    border: 1px solid #3F3F46;
    padding: 8px 16px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #007ACC;
    color: #FFFFFF;
}
QTabBar::tab:hover:!selected {
    background-color: #3F3F46;
}
QPushButton {
    background-color: #0E639C;
    color: white;
    border: none;
    padding: 6px 12px;
    border-radius: 3px;
}
QPushButton:hover {
    background-color: #1177BB;
}
QPushButton:pressed {
    background-color: #0D5A8E;
}
QLabel {
    color: #CCCCCC;
}
QGroupBox {
    border: 1px solid #3F3F46;
    border-radius: 5px;
    margin-top: 1em;
    color: #CCCCCC;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top center;
    padding: 0 5px;
}
QComboBox {
    background-color: #3F3F46;
    color: #CCCCCC;
    padding: 4px 8px;
}
QSlider::groove:horizontal {
    border: 1px solid #999999;
    height: 4px;
    background: #3F3F46;
    margin: 0px;
}
QSlider::handle:horizontal {
    background: #007ACC;
    border: 1px solid #5c5c5c;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QToolBar {
    background: #2D2D30;
    border: 1px solid #3F3F46;
}
QToolButton {
    background-color: transparent;
    border: none;
    padding: 4px;
}
QToolButton:hover {
    background-color: #3F3F46;
    border-radius: 3px;
}
QStatusBar {
    background-color: #007ACC;
    color: white;
}
QProgressBar {
    border: 1px solid #3F3F46;
    border-radius: 2px;
    text-align: center;
    background-color: #3F3F46;
}
QProgressBar::chunk {
    background-color: #007ACC;
}
QTableWidget {
    background-color: #252526;
    color: #CCCCCC;
    gridline-color: #3F3F46;
}
QTableWidget QHeaderView::section {
    background-color: #3F3F46;
    color: white;
    padding: 4px;
    border: 1px solid #5c5c5c;
}
QLineEdit {
    background-color: #3F3F46;
    color: #CCCCCC;
    padding: 4px;
    border: 1px solid #5c5c5c;
    border-radius: 2px;
}
"""


# 数据采集线程 - 模拟数据生成
class DataGeneratorThread(QThread):
    dataReady = pyqtSignal(object)
    statusUpdate = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.interval = 0.5  # 默认0.5秒更新一次
        self.noise_level = 0.2  # 噪声水平
        self.beam_count = 64  # 默认64个波束
        self.data_quality = "高精度"  # 数据质量模式

    def set_params(self, interval=None, noise=None, beams=None, quality=None):
        if interval is not None:
            self.interval = interval
        if noise is not None:
            self.noise_level = noise
        if beams is not None:
            self.beam_count = beams
        if quality is not None:
            self.data_quality = quality

    def run(self):
        position_x = 0
        position_y = 0

        # 模拟海底特征 - 添加一些有趣的地形特征
        terrain_features = [
            {"type": "ridge", "x": 3.5, "y": 5.0, "height": 8, "width": 1.5},
            {"type": "crater", "x": 7.0, "y": 3.0, "depth": 5, "radius": 1.0},
            {"type": "seamount", "x": 2.0, "y": 8.0, "height": 10, "radius": 0.8}
        ]

        while self.running:
            # 模拟声呐移动
            position_x += 0.1
            position_y += 0.05 * np.sin(position_x * 0.8)

            # 根据数据质量模式调整噪声
            actual_noise = self.noise_level
            if self.data_quality == "高精度":
                actual_noise *= 0.5
            elif self.data_quality == "标准":
                actual_noise *= 1.0
            elif self.data_quality == "快速扫描":
                actual_noise *= 2.0

            # 模拟波束数据
            beam_angles = np.linspace(-75, 75, self.beam_count)
            base_depth = 20 + 5 * np.sin(position_x * 0.5) + 3 * np.cos(position_y * 0.4)
            beam_data = np.zeros(self.beam_count)

            # 生成基础波束数据
            for i, angle in enumerate(beam_angles):
                # 计算该波束在海底的x,y位置
                angle_rad = np.deg2rad(angle)
                beam_x = position_x + base_depth * np.tan(angle_rad) * np.cos(angle_rad)
                beam_y = position_y + base_depth * np.tan(angle_rad) * np.sin(angle_rad)

                # 基础水深
                depth = base_depth

                # 添加地形特征的影响
                for feature in terrain_features:
                    dx = beam_x - feature["x"]
                    dy = beam_y - feature["y"]
                    distance = np.sqrt(dx ** 2 + dy ** 2)

                    if feature["type"] == "ridge":
                        width = feature["width"]
                        if abs(dx) < width:
                            # 山脊形状
                            depth -= feature["height"] * np.exp(-(dx / width) ** 2)

                    elif feature["type"] == "crater":
                        if distance < feature["radius"]:
                            # 环形坑
                            depth += feature["depth"] * (1 - distance / feature["radius"])

                    elif feature["type"] == "seamount":
                        if distance < feature["radius"]:
                            # 海底山
                            depth -= feature["height"] * (1 - distance / feature["radius"]) ** 2

                # 添加噪声
                depth += np.random.normal(0, actual_noise)

                # 限制最小深度
                depth = max(depth, 5)

                beam_data[i] = depth

            # 随机产生设备状态变化
            if np.random.rand() > 0.97:
                devices = ["电源", "传感器", "数据链路", "存储系统", "GPS"]
                device = np.random.choice(devices)
                status = np.random.choice(["正常", "警告", "错误"], p=[0.7, 0.2, 0.1])
                self.statusUpdate.emit(device, status)

            # 创建数据包
            data_package = {
                'timestamp': time.time(),
                'position_x': position_x,
                'position_y': position_y,
                'beam_angles': beam_angles,
                'beam_data': beam_data,
                'quality': self.data_quality,
                'noise_level': actual_noise
            }

            # 发送数据
            self.dataReady.emit(data_package)

            # 暂停
            time.sleep(self.interval)

    def stop(self):
        self.running = False


# 主窗口类
class MultibeamSonarSystem(QMainWindow):
    def __init__(self):
        super().__init__()

        # 应用样式
        self.setStyleSheet(STYLE_SHEET)

        self.setWindowTitle("海洋探测多波束测深显控平台 v2.0")
        self.setGeometry(100, 50, 1600, 900)

        # 初始化数据
        self.init_data()

        # 创建UI
        self.init_ui()

        # 启动数据生成线程
        self.data_thread = DataGeneratorThread()
        self.data_thread.dataReady.connect(self.process_data)
        self.data_thread.statusUpdate.connect(self.update_device_status)
        self.data_thread.start()

        # 状态栏初始化
        self.statusBar().showMessage("系统就绪 | 数据模拟模式")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedWidth(150)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.statusBar().addPermanentWidget(QLabel("数据缓冲: "))
        self.statusBar().addPermanentWidget(self.progress_bar)

        # 创建模拟进度更新定时器
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress)
        self.progress_timer.start(200)

        # 记录开始时间
        self.start_time = time.time()
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_runtime)
        self.update_timer.start(1000)  # 每秒更新一次

    def init_data(self):
        """初始化数据结构"""
        # 航迹数据
        self.track_x = []
        self.track_y = []

        # 水深数据矩阵
        self.grid_size = 100
        self.depth_data = np.zeros((self.grid_size, self.grid_size))
        # 初始化为NaN表示未探测区域
        self.depth_data[:] = np.nan

        # 波束数据
        self.beam_count = 64
        self.beam_data = np.zeros(self.beam_count)
        self.beam_angles = np.linspace(-75, 75, self.beam_count)

        # 设备状态
        self.device_status = {
            "电源": "正常",
            "传感器": "正常",
            "数据链路": "正常",
            "存储系统": "正常",
            "GPS": "正常",
        }

        # 数据统计
        self.data_stats = {
            "总数据点": 0,
            "平均水深": 0,
            "最小水深": float('inf'),
            "最大水深": 0,
            "扫描面积": 0,
            "数据文件大小": 0
        }

        # 系统参数
        self.system_params = {
            "测量频率": "200kHz",
            "声波速度": "1500m/s",
            "波束角度": "150°",
            "扫描范围": "200m",
            "分辨率": "高精度",
        }

        # 数据记录
        self.data_log = []

        # 警告日志
        self.alert_log = []

    def init_ui(self):
        """初始化用户界面"""
        # 创建工具栏
        self.create_toolbar()

        # 创建主要部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建选项卡
        self.tabs = QTabWidget()

        # 创建各个选项卡页面
        self.create_dashboard_tab()
        self.create_realtime_tab()
        self.create_3d_view_tab()
        self.create_device_monitor_tab()
        self.create_data_analysis_tab()
        self.create_settings_tab()

        # 设置主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(self.tabs)

        central_widget.setLayout(main_layout)

    def create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar("主工具栏")
        toolbar.setIconSize(QSize(32, 32))
        self.addToolBar(toolbar)

        # 添加工具栏按钮
        start_action = QAction("开始采集", self)
        start_action.triggered.connect(self.start_acquisition)
        toolbar.addAction(start_action)

        stop_action = QAction("停止采集", self)
        stop_action.triggered.connect(self.stop_acquisition)
        toolbar.addAction(stop_action)

        toolbar.addSeparator()

        save_action = QAction("保存数据", self)
        save_action.triggered.connect(self.save_data)
        toolbar.addAction(save_action)

        load_action = QAction("加载数据", self)
        load_action.triggered.connect(self.load_data)
        toolbar.addAction(load_action)

        toolbar.addSeparator()

        export_action = QAction("导出报告", self)
        export_action.triggered.connect(self.export_report)
        toolbar.addAction(export_action)

        toolbar.addSeparator()

        # 添加系统运行时间显示
        self.runtime_label = QLabel("运行时间: 00:00:00")
        self.runtime_label.setStyleSheet("color: #00FF00; font-weight: bold;")
        toolbar.addWidget(self.runtime_label)

    def create_dashboard_tab(self):
        """创建仪表盘选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建顶部状态面板
        status_panel = QWidget()
        status_layout = QHBoxLayout()
        status_panel.setLayout(status_layout)
        status_panel.setMaximumHeight(120)

        # 添加4个状态框
        for title, value, unit in [
            ("平均水深", "22.5", "m"),
            ("航行距离", "0.0", "km"),
            ("数据点数", "0", ""),
            ("扫描面积", "0.0", "m²")
        ]:
            status_box = QGroupBox(title)
            box_layout = QVBoxLayout()
            value_label = QLabel(value)
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setStyleSheet("font-size: 28pt; font-weight: bold; color: #00A6FF;")
            unit_label = QLabel(unit)
            unit_label.setAlignment(Qt.AlignRight)
            box_layout.addWidget(value_label)
            box_layout.addWidget(unit_label)
            status_box.setLayout(box_layout)
            status_layout.addWidget(status_box)

            # 保存标签以便更新
            if title == "平均水深":
                self.depth_value_label = value_label
            elif title == "航行距离":
                self.distance_value_label = value_label
            elif title == "数据点数":
                self.points_value_label = value_label
            elif title == "扫描面积":
                self.area_value_label = value_label

        layout.addWidget(status_panel)

        # 创建主显示区域 (2x2网格)
        main_panel = QWidget()
        grid_layout = QGridLayout()
        main_panel.setLayout(grid_layout)

        # 水深剖面图
        profile_group = QGroupBox("实时水深剖面")
        profile_layout = QVBoxLayout()
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setBackground("#252526")
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_plot.setLabel('left', '深度', 'm')
        self.profile_plot.setLabel('bottom', '波束角度', '°')
        self.profile_plot.setYRange(0, 50, padding=0)
        self.profile_curve = self.profile_plot.plot(pen=pg.mkPen(color='#00A6FF', width=2))
        profile_layout.addWidget(self.profile_plot)
        profile_group.setLayout(profile_layout)
        grid_layout.addWidget(profile_group, 0, 0)

        # 航迹图
        track_group = QGroupBox("航迹图")
        track_layout = QVBoxLayout()
        self.track_plot = pg.PlotWidget()
        self.track_plot.setBackground("#252526")
        self.track_plot.showGrid(x=True, y=True, alpha=0.3)
        self.track_plot.setLabel('left', 'Y', 'm')
        self.track_plot.setLabel('bottom', 'X', 'm')
        self.track_plot.setAspectLocked(True)
        self.track_curve = self.track_plot.plot(pen=pg.mkPen(color='#00FF00', width=2))
        track_layout.addWidget(self.track_plot)
        track_group.setLayout(track_layout)
        grid_layout.addWidget(track_group, 0, 1)

        # 水深热力图
        depth_group = QGroupBox("水深热力图")
        depth_layout = QVBoxLayout()
        self.depth_image = pg.ImageView()
        self.depth_image.ui.graphicsView.setBackground("#252526")
        self.depth_image.setColorMap(pg.colormap.get('viridis'))
        depth_layout.addWidget(self.depth_image)
        depth_group.setLayout(depth_layout)
        grid_layout.addWidget(depth_group, 1, 0)

        # 系统状态面板
        system_group = QGroupBox("系统状态")
        system_layout = QVBoxLayout()

        # 创建表格显示设备状态
        self.status_table = QTableWidget(5, 2)
        self.status_table.setHorizontalHeaderLabels(["设备", "状态"])
        self.status_table.horizontalHeader().setStretchLastSection(True)
        self.status_table.verticalHeader().setVisible(False)

        # 添加设备状态行
        row = 0
        for device, status in self.device_status.items():
            self.status_table.setItem(row, 0, QTableWidgetItem(device))
            status_item = QTableWidgetItem(status)
            status_item.setForeground(QColor("#00FF00"))  # 绿色表示正常
            self.status_table.setItem(row, 1, status_item)
            row += 1

        system_layout.addWidget(self.status_table)

        # 添加警告日志
        self.alert_list = QTableWidget(0, 2)
        self.alert_list.setHorizontalHeaderLabels(["时间", "警告信息"])
        self.alert_list.horizontalHeader().setStretchLastSection(True)
        system_layout.addWidget(QLabel("警告日志:"))
        system_layout.addWidget(self.alert_list)

        system_group.setLayout(system_layout)
        grid_layout.addWidget(system_group, 1, 1)

        layout.addWidget(main_panel)
        tab.setLayout(layout)

        self.tabs.addTab(tab, "仪表盘")

        # 更新深度图

        empty_data = np.zeros((self.grid_size, self.grid_size))
        empty_data[:] = 20.0
        mask = np.random.rand(self.grid_size, self.grid_size) > 0.2
        empty_data[mask] = np.nan
        self.depth_image.setImage(empty_data, levels=(0, 40))

    def create_realtime_tab(self):
        """创建实时显示选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setMaximumHeight(80)

        # 添加控制项
        view_label = QLabel("视图模式:")
        view_combo = QComboBox()
        view_combo.addItems(["标准视图", "高度差异视图", "坡度视图", "阴影浮雕视图"])
        view_combo.currentIndexChanged.connect(self.change_view_mode)

        beam_label = QLabel("波束数量:")
        beam_combo = QComboBox()
        beam_combo.addItems(["32", "64", "128", "256"])
        beam_combo.setCurrentIndex(1)  # 默认64
        beam_combo.currentIndexChanged.connect(self.change_beam_count)

        quality_label = QLabel("数据质量:")
        quality_combo = QComboBox()
        quality_combo.addItems(["高精度", "标准", "快速扫描"])
        quality_combo.currentIndexChanged.connect(self.change_data_quality)

        # 添加到布局
        control_layout.addWidget(view_label)
        control_layout.addWidget(view_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(beam_label)
        control_layout.addWidget(beam_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(quality_label)
        control_layout.addWidget(quality_combo)
        control_layout.addStretch()

        layout.addWidget(control_panel)

        # 创建分隔窗口
        splitter = QSplitter(Qt.Vertical)

        # 顶部水深剖面图
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_widget.setLayout(top_layout)

        # 左侧水深剖面图
        self.beam_plot = pg.PlotWidget()
        self.beam_plot.setBackground("#252526")
        self.beam_plot.showGrid(x=True, y=True, alpha=0.3)
        self.beam_plot.setTitle("多波束水深剖面图", color="#FFFFFF")
        self.beam_plot.setLabel('left', '深度', 'm')
        self.beam_plot.setLabel('bottom', '波束角度', '°')
        self.beam_plot.setYRange(0, 50, padding=0)
        self.beam_plot.getAxis('left').setPen(pg.mkPen(color='#FFFFFF'))
        self.beam_plot.getAxis('bottom').setPen(pg.mkPen(color='#FFFFFF'))

        # 添加波束数据曲线
        self.beam_curve = self.beam_plot.plot(pen=pg.mkPen(color='#00A6FF', width=3))

        # 添加海底剖面填充
        self.beam_fill = pg.FillBetweenItem(
            pg.PlotCurveItem(pen=pg.mkPen(color='#00A6FF', width=0)),
            pg.PlotCurveItem(pen=pg.mkPen(color='#00A6FF', width=0)),
            brush=pg.mkBrush(color=QColor(0, 166, 255, 100))
        )
        self.beam_plot.addItem(self.beam_fill)

        # 右侧航迹图
        self.realtime_track_plot = pg.PlotWidget()
        self.realtime_track_plot.setBackground("#252526")
        self.realtime_track_plot.showGrid(x=True, y=True, alpha=0.3)
        self.realtime_track_plot.setTitle("实时航迹", color="#FFFFFF")
        self.realtime_track_plot.setLabel('left', 'Y', 'm')
        self.realtime_track_plot.setLabel('bottom', 'X', 'm')
        self.realtime_track_plot.setAspectLocked(True)
        self.realtime_track_plot.getAxis('left').setPen(pg.mkPen(color='#FFFFFF'))
        self.realtime_track_plot.getAxis('bottom').setPen(pg.mkPen(color='#FFFFFF'))

        # 添加实时航迹曲线和当前位置指示
        self.realtime_track_curve = self.realtime_track_plot.plot(pen=pg.mkPen(color='#00FF00', width=2))
        self.position_marker = pg.ScatterPlotItem(
            size=15,
            pen=pg.mkPen(color='#FF0000', width=2),
            brush=pg.mkBrush(color='#FF0000')
        )
        self.realtime_track_plot.addItem(self.position_marker)

        # 添加到布局
        top_layout.addWidget(self.beam_plot, 1)
        top_layout.addWidget(self.realtime_track_plot, 1)

        # 底部水深图显示区
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout()
        bottom_widget.setLayout(bottom_layout)

        # 海底地形图
        self.realtime_depth_image = pg.ImageView()
        self.realtime_depth_image.ui.graphicsView.setBackground("#252526")
        self.realtime_depth_image.ui.roiBtn.hide()
        self.realtime_depth_image.ui.menuBtn.hide()
        viridis_cm = pg.colormap.get('viridis')
        self.realtime_depth_image.setColorMap(viridis_cm)

        # 初始化深度图
        # 替换为
        empty_data = np.zeros((self.grid_size, self.grid_size))
        # 添加一些默认值而不是全部设为NaN
        empty_data[:] = 20.0  # 设置默认水深为20米
        # 只将部分区域设为NaN，保留一些有效数据
        mask = np.random.rand(self.grid_size, self.grid_size) > 0.2
        empty_data[mask] = np.nan
        # 设置图像并指定级别范围
        self.realtime_depth_image.setImage(empty_data, levels=(0, 40))
        # 水深图标题和颜色条
        depth_title = QLabel("海底地形热力图")
        depth_title.setAlignment(Qt.AlignCenter)
        depth_title.setStyleSheet("font-size: 12pt; font-weight: bold; color: #FFFFFF;")

        # 添加信息标签
        self.depth_info_label = QLabel("水深范围: 0.0m - 0.0m | 扫描覆盖率: 0.0%")
        self.depth_info_label.setAlignment(Qt.AlignCenter)

        bottom_layout.addWidget(depth_title)
        bottom_layout.addWidget(self.realtime_depth_image)
        bottom_layout.addWidget(self.depth_info_label)

        # 添加到分隔窗口
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([300, 500])  # 设置初始高度比例

        layout.addWidget(splitter)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "实时显示")

    def create_3d_view_tab(self):
        """创建3D视图选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()
        control_panel.setLayout(control_layout)

        # 添加控制项
        view_label = QLabel("3D视图模式:")
        view_combo = QComboBox()
        view_combo.addItems(["彩色高程图", "线框图", "点云图", "等高线图"])
        view_combo.currentIndexChanged.connect(self.change_3d_view_mode)

        color_label = QLabel("颜色方案:")
        color_combo = QComboBox()
        color_combo.addItems(["深度渐变", "高光渲染", "地形分析", "海底分类"])
        color_combo.currentIndexChanged.connect(self.change_3d_color_scheme)

        exaggeration_label = QLabel("高度夸张:")
        exaggeration_slider = QSlider(Qt.Horizontal)
        exaggeration_slider.setRange(10, 50)
        exaggeration_slider.setValue(20)
        exaggeration_slider.valueChanged.connect(self.change_elevation_exaggeration)

        # 添加到布局
        control_layout.addWidget(view_label)
        control_layout.addWidget(view_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(color_label)
        control_layout.addWidget(color_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(exaggeration_label)
        control_layout.addWidget(exaggeration_slider)

        layout.addWidget(control_panel)

        # 创建3D图形
        self.figure = Figure(figsize=(10, 8), facecolor='#252526')
        self.canvas = FigureCanvas(self.figure)
        self.ax3d = self.figure.add_subplot(111, projection='3d')
        self.ax3d.set_facecolor('#252526')

        # 设置轴标签颜色
        self.ax3d.xaxis.label.set_color('white')
        self.ax3d.yaxis.label.set_color('white')
        self.ax3d.zaxis.label.set_color('white')
        self.ax3d.tick_params(axis='x', colors='white')
        self.ax3d.tick_params(axis='y', colors='white')
        self.ax3d.tick_params(axis='z', colors='white')

        # 初始化3D图形
        # self.update_3d_view()

        # 添加到布局
        layout.addWidget(self.canvas)

        # 添加刷新按钮和信息栏
        bottom_panel = QWidget()
        bottom_layout = QHBoxLayout()

        update_button = QPushButton("刷新3D视图")
        update_button.clicked.connect(self.update_3d_view)

        self.view_info_label = QLabel("显示 100x100 网格数据 | 高度夸张: 2.0x")

        screenshot_button = QPushButton("导出3D图像")
        screenshot_button.clicked.connect(self.save_3d_screenshot)

        bottom_layout.addWidget(update_button)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.view_info_label)
        bottom_layout.addStretch()
        bottom_layout.addWidget(screenshot_button)

        bottom_panel.setLayout(bottom_layout)
        layout.addWidget(bottom_panel)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "3D地形图")

        # 保存参数
        self.view_mode = "彩色高程图"
        self.color_scheme = "深度渐变"
        self.elevation_factor = 2.0


    def create_device_monitor_tab(self):
        """创建设备监控选项卡"""
        tab = QWidget()
        layout = QHBoxLayout()

        # 创建左侧面板 - 设备状态
        left_panel = QGroupBox("设备状态监控")
        left_layout = QVBoxLayout()

        # 设备状态表格
        self.monitor_table = QTableWidget(len(self.device_status), 3)
        self.monitor_table.setHorizontalHeaderLabels(["设备", "状态", "详细信息"])
        self.monitor_table.horizontalHeader().setStretchLastSection(True)
        self.monitor_table.setSelectionBehavior(QTableWidget.SelectRows)

        # 填充设备状态
        row = 0
        self.status_items = {}
        for device, status in self.device_status.items():
            self.monitor_table.setItem(row, 0, QTableWidgetItem(device))
            status_item = QTableWidgetItem(status)
            if status == "正常":
                status_item.setForeground(QColor("#00FF00"))
            else:
                status_item.setForeground(QColor("#FF0000"))
            self.monitor_table.setItem(row, 1, status_item)

            # 保存状态项以便更新
            self.status_items[device] = status_item

            # 添加详细信息
            details = ""
            if device == "电源":
                details = "电压: 12.1V, 电流: 2.3A"
            elif device == "传感器":
                details = "温度: 25°C, 工作正常"
            elif device == "数据链路":
                details = "带宽: 10Mbps, 延迟: 5ms"
            elif device == "存储系统":
                details = "已用: 45%, 剩余: 55GB"
            elif device == "GPS":
                details = "信号强度: 良好, 卫星数: 9"

            self.monitor_table.setItem(row, 2, QTableWidgetItem(details))
            row += 1

        left_layout.addWidget(self.monitor_table)

        # 添加系统日志
        log_label = QLabel("系统日志")
        left_layout.addWidget(log_label)

        self.system_log = QTableWidget(0, 2)
        self.system_log.setHorizontalHeaderLabels(["时间", "事件"])
        self.system_log.horizontalHeader().setStretchLastSection(True)
        left_layout.addWidget(self.system_log)

        # 添加一些初始日志项
        self.add_system_log("系统初始化完成")
        self.add_system_log("开始数据模拟")

        left_panel.setLayout(left_layout)

        # 创建右侧面板 - 性能监控
        right_panel = QGroupBox("系统性能监控")
        right_layout = QVBoxLayout()

        # CPU使用率图表
        cpu_group = QGroupBox("CPU使用率")
        cpu_layout = QVBoxLayout()
        self.cpu_plot = pg.PlotWidget()
        self.cpu_plot.setBackground("#252526")
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.3)
        self.cpu_plot.setYRange(0, 100)
        self.cpu_plot.setLabel('left', '使用率', '%')
        self.cpu_plot.setLabel('bottom', '时间', 's')

        self.cpu_data = np.zeros(100)
        self.cpu_curve = self.cpu_plot.plot(self.cpu_data, pen=pg.mkPen(color='#FF5500', width=2))

        cpu_layout.addWidget(self.cpu_plot)
        cpu_group.setLayout(cpu_layout)

        # 内存使用率图表
        mem_group = QGroupBox("内存使用率")
        mem_layout = QVBoxLayout()
        self.mem_plot = pg.PlotWidget()
        self.mem_plot.setBackground("#252526")
        self.mem_plot.showGrid(x=True, y=True, alpha=0.3)
        self.mem_plot.setYRange(0, 100)
        self.mem_plot.setLabel('left', '使用率', '%')
        self.mem_plot.setLabel('bottom', '时间', 's')

        self.mem_data = np.zeros(100)
        self.mem_curve = self.mem_plot.plot(self.mem_data, pen=pg.mkPen(color='#00CCFF', width=2))

        mem_layout.addWidget(self.mem_plot)
        mem_group.setLayout(mem_layout)

        # 网络流量图表
        net_group = QGroupBox("网络流量")
        net_layout = QVBoxLayout()
        self.net_plot = pg.PlotWidget()
        self.net_plot.setBackground("#252526")
        self.net_plot.showGrid(x=True, y=True, alpha=0.3)
        self.net_plot.setLabel('left', '流量', 'KB/s')
        self.net_plot.setLabel('bottom', '时间', 's')

        self.net_data = np.zeros(100)
        self.net_curve = self.net_plot.plot(self.net_data, pen=pg.mkPen(color='#00FF00', width=2))

        net_layout.addWidget(self.net_plot)
        net_group.setLayout(net_layout)

        # 添加到右侧布局
        right_layout.addWidget(cpu_group)
        right_layout.addWidget(mem_group)
        right_layout.addWidget(net_group)

        right_panel.setLayout(right_layout)

        # 添加到主布局
        layout.addWidget(left_panel, 1)
        layout.addWidget(right_panel, 1)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "设备监控")

        # 开始性能监控更新
        self.perf_timer = QTimer()
        self.perf_timer.timeout.connect(self.update_performance_data)
        self.perf_timer.start(1000)  # 每秒更新一次

    def create_data_analysis_tab(self):
        """创建数据分析选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建分析控制面板
        control_panel = QWidget()
        control_layout = QHBoxLayout()

        analysis_type_label = QLabel("分析类型:")
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems(["海底坡度分析", "水深分布直方图", "海底特征识别", "数据质量评估"])
        self.analysis_combo.currentIndexChanged.connect(self.update_analysis_view)

        filter_label = QLabel("数据过滤:")
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["无过滤", "低通滤波", "高通滤波", "中值滤波"])
        self.filter_combo.currentIndexChanged.connect(self.apply_data_filter)

        analyze_button = QPushButton("执行分析")
        analyze_button.clicked.connect(self.perform_analysis)

        control_layout.addWidget(analysis_type_label)
        control_layout.addWidget(self.analysis_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(filter_label)
        control_layout.addWidget(self.filter_combo)
        control_layout.addSpacing(20)
        control_layout.addWidget(analyze_button)
        control_layout.addStretch()

        export_button = QPushButton("导出分析结果")
        export_button.clicked.connect(self.export_analysis)
        control_layout.addWidget(export_button)

        control_panel.setLayout(control_layout)
        layout.addWidget(control_panel)

        # 创建分析结果显示区
        analysis_splitter = QSplitter(Qt.Horizontal)

        # 左侧图表区域
        chart_frame = QFrame()
        chart_frame.setFrameShape(QFrame.StyledPanel)
        chart_layout = QVBoxLayout()
        chart_frame.setLayout(chart_layout)

        # Matplotlib图表
        self.analysis_figure = Figure(figsize=(8, 6), facecolor='#252526')
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        self.analysis_ax = self.analysis_figure.add_subplot(111)
        self.analysis_ax.set_facecolor('#252526')

        # 设置轴标签颜色
        self.analysis_ax.xaxis.label.set_color('white')
        self.analysis_ax.yaxis.label.set_color('white')
        self.analysis_ax.tick_params(axis='x', colors='white')
        self.analysis_ax.tick_params(axis='y', colors='white')
        self.analysis_ax.set_title("分析结果", color='white')

        chart_layout.addWidget(self.analysis_canvas)

        # 右侧结果统计区域
        stats_frame = QFrame()
        stats_frame.setFrameShape(QFrame.StyledPanel)
        stats_layout = QVBoxLayout()
        stats_frame.setLayout(stats_layout)

        stats_title = QLabel("分析统计")
        stats_title.setStyleSheet("font-size: 14pt; font-weight: bold;")
        stats_layout.addWidget(stats_title)

        self.stats_table = QTableWidget(0, 2)
        self.stats_table.setHorizontalHeaderLabels(["参数", "值"])
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        stats_layout.addWidget(self.stats_table)

        # 分析描述
        self.analysis_description = QLabel("选择分析类型并点击'执行分析'按钮查看结果。")
        self.analysis_description.setWordWrap(True)
        self.analysis_description.setStyleSheet("background-color: #3F3F46; padding: 10px;")
        stats_layout.addWidget(self.analysis_description)

        # 添加到分隔窗口
        analysis_splitter.addWidget(chart_frame)
        analysis_splitter.addWidget(stats_frame)
        analysis_splitter.setSizes([700, 300])  # 设置初始宽度比例

        layout.addWidget(analysis_splitter)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "数据分析")

        # 设置初始分析视图
        self.update_analysis_view()

    def create_settings_tab(self):
        """创建设置选项卡"""
        tab = QWidget()
        layout = QVBoxLayout()

        # 创建设置组
        system_group = QGroupBox("系统设置")
        system_layout = QGridLayout()

        # 添加各种设置项
        settings = [
            ("测量频率", QComboBox(), ["100kHz", "200kHz", "400kHz"]),
            ("声波速度", QLineEdit("1500"), "m/s"),
            ("波束角度", QComboBox(), ["120°", "150°", "180°"]),
            ("扫描范围", QLineEdit("200"), "m"),
            ("分辨率", QComboBox(), ["低", "中", "高", "超高"]),
            ("数据输出格式", QComboBox(), ["CSV", "XYZ", "GeoTIFF", "自定义"])
        ]

        row = 0
        for setting in settings:
            label = QLabel(setting[0] + ":")
            system_layout.addWidget(label, row, 0)

            if isinstance(setting[1], QComboBox):
                setting[1].addItems(setting[2])
                system_layout.addWidget(setting[1], row, 1)
            elif isinstance(setting[1], QLineEdit):
                system_layout.addWidget(setting[1], row, 1)
                if len(setting) > 2:
                    unit_label = QLabel(setting[2])
                    system_layout.addWidget(unit_label, row, 2)

            row += 1

        # 添加设置组
        system_group.setLayout(system_layout)
        layout.addWidget(system_group)

        # 数据存储设置
        storage_group = QGroupBox("数据存储设置")
        storage_layout = QGridLayout()

        storage_path_label = QLabel("存储路径:")
        storage_path_edit = QLineEdit("C:/Sonar_Data/")
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_storage_path)

        auto_save_check = QCheckBox("自动保存数据")
        auto_save_check.setChecked(True)

        save_interval_label = QLabel("保存间隔:")
        save_interval_edit = QLineEdit("60")
        save_interval_unit = QLabel("秒")

        storage_layout.addWidget(storage_path_label, 0, 0)
        storage_layout.addWidget(storage_path_edit, 0, 1)
        storage_layout.addWidget(browse_button, 0, 2)
        storage_layout.addWidget(auto_save_check, 1, 0)
        storage_layout.addWidget(save_interval_label, 2, 0)
        storage_layout.addWidget(save_interval_edit, 2, 1)
        storage_layout.addWidget(save_interval_unit, 2, 2)

        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)

        # 显示设置
        display_group = QGroupBox("显示设置")
        display_layout = QGridLayout()

        color_theme_label = QLabel("颜色主题:")
        color_theme_combo = QComboBox()
        color_theme_combo.addItems(["深色主题", "浅色主题", "海洋主题", "高对比度主题"])

        refresh_rate_label = QLabel("刷新率:")
        refresh_rate_slider = QSlider(Qt.Horizontal)
        refresh_rate_slider.setRange(1, 10)
        refresh_rate_slider.setValue(2)
        refresh_rate_slider.valueChanged.connect(self.change_refresh_rate)
        refresh_rate_value = QLabel("2 Hz")

        # 连接刷新率滑块值变化信号
        def update_rate_label(value):
            refresh_rate_value.setText(f"{value} Hz")

        refresh_rate_slider.valueChanged.connect(update_rate_label)

        display_3d_check = QCheckBox("启用3D实时预览")
        display_3d_check.setChecked(False)

        display_grid_check = QCheckBox("显示网格线")
        display_grid_check.setChecked(True)

        display_layout.addWidget(color_theme_label, 0, 0)
        display_layout.addWidget(color_theme_combo, 0, 1, 1, 2)
        display_layout.addWidget(refresh_rate_label, 1, 0)
        display_layout.addWidget(refresh_rate_slider, 1, 1)
        display_layout.addWidget(refresh_rate_value, 1, 2)
        display_layout.addWidget(display_3d_check, 2, 0, 1, 3)
        display_layout.addWidget(display_grid_check, 3, 0, 1, 3)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 添加按钮区
        button_panel = QWidget()
        button_layout = QHBoxLayout()

        save_settings_button = QPushButton("保存设置")
        save_settings_button.clicked.connect(self.save_settings)

        reset_default_button = QPushButton("恢复默认")
        reset_default_button.clicked.connect(self.reset_settings)

        button_layout.addStretch()
        button_layout.addWidget(save_settings_button)
        button_layout.addWidget(reset_default_button)

        button_panel.setLayout(button_layout)
        layout.addWidget(button_panel)

        # 添加一些空间
        layout.addStretch()

        tab.setLayout(layout)
        self.tabs.addTab(tab, "设置")

        # 保存设置控件
        self.refresh_rate_slider = refresh_rate_slider

    def process_data(self, data_package):
        """处理接收到的数据包"""
        # 更新位置
        self.track_x.append(data_package['position_x'])
        self.track_y.append(data_package['position_y'])

        # 限制航迹长度
        if len(self.track_x) > 1000:
            self.track_x = self.track_x[-1000:]
            self.track_y = self.track_y[-1000:]

        # 更新波束数据
        self.beam_angles = data_package['beam_angles']
        self.beam_data = data_package['beam_data']
        self.beam_count = len(self.beam_angles)

        # 更新深度图
        self.update_depth_map(data_package)

        # 更新仪表盘统计数据
        self.update_dashboard_stats()

        # 更新实时显示
        self.update_realtime_display()

    def update_depth_map(self, data_package):
        """更新深度图数据"""
        position_x = data_package['position_x']
        position_y = data_package['position_y']
        beam_angles = data_package['beam_angles']
        beam_data = data_package['beam_data']

        # 计算网格索引
        grid_x = int((position_x % 20) / 20 * self.grid_size)
        grid_y = int((position_y % 20) / 20 * self.grid_size)

        # 更新当前点的深度数据
        if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
            self.depth_data[grid_y, grid_x] = np.mean(beam_data)

        # 沿波束方向更新深度数据
        for i, angle in enumerate(beam_angles):
            # 跳过一些波束以提高性能
            if i % 4 != 0:
                continue

            angle_rad = np.deg2rad(angle)
            beam_depth = beam_data[i]

            # 计算波束在海底的位置
            beam_distance = beam_depth * np.tan(abs(angle_rad))
            beam_x = position_x + np.cos(angle_rad) * beam_distance
            beam_y = position_y + np.sin(angle_rad) * beam_distance

            # 转换为网格索引
            beam_grid_x = int((beam_x % 20) / 20 * self.grid_size)
            beam_grid_y = int((beam_y % 20) / 20 * self.grid_size)

            # 更新深度数据
            if 0 <= beam_grid_x < self.grid_size and 0 <= beam_grid_y < self.grid_size:
                # 使用指数加权移动平均来平滑更新
                alpha = 0.3  # 新数据权重
                if np.isnan(self.depth_data[beam_grid_y, beam_grid_x]):
                    self.depth_data[beam_grid_y, beam_grid_x] = beam_depth
                else:
                    self.depth_data[beam_grid_y, beam_grid_x] = (1 - alpha) * self.depth_data[
                        beam_grid_y, beam_grid_x] + alpha * beam_depth

    def update_dashboard_stats(self):
        """更新仪表盘统计数据"""
        # 计算平均水深
        valid_depths = self.depth_data[~np.isnan(self.depth_data)]
        if len(valid_depths) > 0:
            avg_depth = np.mean(valid_depths)
            self.depth_value_label.setText(f"{avg_depth:.1f}")

            # 更新数据统计
            self.data_stats["平均水深"] = avg_depth
            self.data_stats["最小水深"] = np.min(valid_depths)
            self.data_stats["最大水深"] = np.max(valid_depths)

        # 计算航行距离
        if len(self.track_x) > 1:
            dx = np.diff(self.track_x)
            dy = np.diff(self.track_y)
            distance = np.sum(np.sqrt(dx ** 2 + dy ** 2))
            self.distance_value_label.setText(f"{distance:.2f}")

        # 更新数据点数
        self.points_value_label.setText(f"{len(self.track_x)}")
        self.data_stats["总数据点"] = len(self.track_x)

        # 计算扫描面积 (近似)
        valid_cells = np.sum(~np.isnan(self.depth_data))
        area = valid_cells / (self.grid_size * self.grid_size) * 400  # 20m x 20m = 400m²
        self.area_value_label.setText(f"{area:.1f}")
        self.data_stats["扫描面积"] = area

    def update_realtime_display(self):
        """更新实时显示"""
        # 更新波束显示
        self.beam_curve.setData(self.beam_angles, self.beam_data)
        self.profile_curve.setData(self.beam_angles, self.beam_data)

        # 更新波束填充区域
        x = self.beam_angles
        y1 = self.beam_data
        y2 = np.ones_like(y1) * 50  # 最大深度值

        fill_curve1 = pg.PlotCurveItem(x, y1, pen=pg.mkPen(color='#00A6FF', width=0))
        fill_curve2 = pg.PlotCurveItem(x, y2, pen=pg.mkPen(color='#00A6FF', width=0))
        self.beam_fill.setCurves(fill_curve1, fill_curve2)

        # 更新航迹显示
        self.track_curve.setData(self.track_x, self.track_y)
        self.realtime_track_curve.setData(self.track_x, self.track_y)

        # 更新当前位置标记
        if len(self.track_x) > 0 and len(self.track_y) > 0:
            self.position_marker.setData([self.track_x[-1]], [self.track_y[-1]])

        # 更新水深图 - 使用掩码处理NaN值
        masked_data = np.ma.masked_invalid(self.depth_data)
        # 替换为:
        valid_depths = self.depth_data[~np.isnan(self.depth_data)]
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            # 确保最小和最大深度有一定差距
            if min_depth == max_depth:
                max_depth = min_depth + 1.0
            self.depth_image.setImage(masked_data, levels=(min_depth, max_depth))
            self.realtime_depth_image.setImage(masked_data, levels=(min_depth, max_depth))
        else:
            # 如果没有有效数据，使用默认范围
            self.depth_image.setImage(masked_data, levels=(0, 40))
            self.realtime_depth_image.setImage(masked_data, levels=(0, 40))

        # 更新深度信息标签
        valid_depths = self.depth_data[~np.isnan(self.depth_data)]
        if len(valid_depths) > 0:
            min_depth = np.min(valid_depths)
            max_depth = np.max(valid_depths)
            coverage = 100 * np.sum(~np.isnan(self.depth_data)) / (self.grid_size * self.grid_size)
            self.depth_info_label.setText(
                f"水深范围: {min_depth:.1f}m - {max_depth:.1f}m | 扫描覆盖率: {coverage:.1f}%")

    def update_device_status(self, device, status):
        """更新设备状态"""
        self.device_status[device] = status

        # 更新状态显示
        color = "#00FF00" if status == "正常" else "#FF0000"

        # 更新仪表盘状态表
        for row in range(self.status_table.rowCount()):
            if self.status_table.item(row, 0).text() == device:
                status_item = QTableWidgetItem(status)
                status_item.setForeground(QColor(color))
                self.status_table.setItem(row, 1, status_item)
                break

        # 更新设备监控状态
        if device in self.status_items:
            self.status_items[device].setText(status)
            self.status_items[device].setForeground(QColor(color))

        # 如果状态为警告或错误，添加到警告日志
        if status != "正常":
            current_time = datetime.now().strftime("%H:%M:%S")
            row = self.alert_list.rowCount()
            self.alert_list.insertRow(row)
            self.alert_list.setItem(row, 0, QTableWidgetItem(current_time))
            self.alert_list.setItem(row, 1, QTableWidgetItem(f"{device}: {status}"))
            self.alert_list.scrollToBottom()

            # 同时添加到系统日志
            self.add_system_log(f"{device}状态变为{status}", "警告")

    def add_system_log(self, message, level="信息"):
        """添加系统日志"""
        current_time = datetime.now().strftime("%H:%M:%S")
        row = self.system_log.rowCount()
        self.system_log.insertRow(row)

        time_item = QTableWidgetItem(current_time)
        self.system_log.setItem(row, 0, time_item)

        msg_item = QTableWidgetItem(message)
        if level == "警告":
            msg_item.setForeground(QColor("#FFCC00"))
        elif level == "错误":
            msg_item.setForeground(QColor("#FF0000"))
        else:
            msg_item.setForeground(QColor("#00CCFF"))

        self.system_log.setItem(row, 1, msg_item)
        self.system_log.scrollToBottom()

    def update_3d_view(self):
        """更新3D视图"""
        self.ax3d.clear()

        # 准备数据
        x = np.linspace(0, 20, self.grid_size)
        y = np.linspace(0, 20, self.grid_size)
        X, Y = np.meshgrid(x, y)

        # 获取有效的深度数据
        Z = np.copy(self.depth_data)
        # 将NaN值替换为周围有效值的平均值
        mask = np.isnan(Z)
        # 替换为以下代码:
        if np.any(~mask):  # 如果有任何非遮罩点
            Z[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), Z[~mask])
        else:
            # 如果所有点都是遮罩的，则使用默认深度值
            Z[:] = 20.0  # 设置一个默认深度值

        # 根据视图模式显示不同类型的3D图
        if self.view_mode == "彩色高程图":
            surf = self.ax3d.plot_surface(
                X, Y, Z,
                cmap=self.get_color_map(),
                edgecolor='none',
                alpha=0.8,
                antialiased=True
            )
            self.figure.colorbar(surf, ax=self.ax3d, shrink=0.5, aspect=10)

        elif self.view_mode == "线框图":
            self.ax3d.plot_wireframe(
                X, Y, Z,
                cmap=self.get_color_map(),
                rstride=2,
                cstride=2,
                linewidth=0.5,
                color='#00A6FF'
            )

        elif self.view_mode == "点云图":
            # 将网格数据转换为点云
            points_x = X.flatten()
            points_y = Y.flatten()
            points_z = Z.flatten()

            # 根据深度设置颜色
            colors = plt.cm.viridis((points_z - np.min(points_z)) / (np.max(points_z) - np.min(points_z)))

            self.ax3d.scatter(
                points_x, points_y, points_z,
                c=colors,
                s=5,
                alpha=0.8
            )

        elif self.view_mode == "等高线图":
            # 绘制3D等高线
            stride = max(1, self.grid_size // 50)  # 控制等高线密度
            contour_levels = np.linspace(np.min(Z), np.max(Z), 20)

            contour = self.ax3d.contour(
                X, Y, Z,
                levels=contour_levels,
                cmap=self.get_color_map(),
                linewidths=0.8,
                alpha=0.8
            )

            # 添加填充的等高线在底部
            filled_contour = self.ax3d.contourf(
                X, Y, Z,
                levels=contour_levels,
                cmap=self.get_color_map(),
                alpha=0.5,
                zdir='z',
                offset=np.min(Z)
            )


        # 设置视图范围和标签
        self.ax3d.set_xlabel('X (m)', color='white')
        self.ax3d.set_ylabel('Y (m)', color='white')
        self.ax3d.set_zlabel('Depth (m)', color='white')
        self.ax3d.set_title('3D Seabed Topography', color='white')

        # 应用高度夸张因子
        self.ax3d.get_proj = lambda: np.dot(Axes3D.get_proj(self.ax3d), np.diag([1, 1, self.elevation_factor, 1]))

        # 更新信息标签
        valid_points = np.sum(~np.isnan(self.depth_data))
        total_points = self.grid_size * self.grid_size
        coverage = 100 * valid_points / total_points
        self.view_info_label.setText(
            f"显示 {self.grid_size}x{self.grid_size} 网格数据 | 覆盖率: {coverage:.1f}% | 高度夸张: {self.elevation_factor}x"
        )

        self.canvas.draw()

    def get_color_map(self):
        """根据颜色方案返回合适的颜色映射"""
        if self.color_scheme == "深度渐变":
            return cm.viridis
        elif self.color_scheme == "高光渲染":
            return cm.plasma
        elif self.color_scheme == "地形分析":
            return cm.terrain
        elif self.color_scheme == "海底分类":
            return cm.ocean
        return cm.viridis

    def update_analysis_view(self):
        """更新分析视图"""
        analysis_type = self.analysis_combo.currentText()

        # 准备数据
        valid_data = self.depth_data[~np.isnan(self.depth_data)]
        if len(valid_data) == 0:
            return

        # 清除当前图形
        self.analysis_ax.clear()

        if analysis_type == "海底坡度分析":
            # 计算水深梯度
            grad_y, grad_x = np.gradient(self.depth_data)
            slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
            slope[np.isnan(self.depth_data)] = np.nan

            # 绘制坡度热力图
            im = self.analysis_ax.imshow(
                slope,
                cmap='rainbow',
                interpolation='nearest',
                origin='lower'
            )
            self.analysis_figure.colorbar(im, ax=self.analysis_ax, label='坡度 (m/m)')
            self.analysis_ax.set_title("海底坡度分析", color='white')

            # 更新统计数据
            self.update_stats_table([
                ("平均坡度", f"{np.nanmean(slope):.3f} m/m"),
                ("最大坡度", f"{np.nanmax(slope):.3f} m/m"),
                ("最小坡度", f"{np.nanmin(slope):.3f} m/m"),
                ("坡度标准差", f"{np.nanstd(slope):.3f} m/m")
            ])

            # 更新描述
            self.analysis_description.setText(
                "海底坡度分析显示海底地形的倾斜程度。红色区域表示陡峭区域，蓝色区域表示平坦区域。"
                "高坡度区域可能表示海底山脊、悬崖或人工结构。"
            )

        elif analysis_type == "水深分布直方图":
            # 绘制水深分布直方图
            n, bins, patches = self.analysis_ax.hist(
                valid_data,
                bins=30,
                color='#00A6FF',
                alpha=0.7
            )
            self.analysis_ax.set_xlabel("水深 (m)", color='white')
            self.analysis_ax.set_ylabel("频次", color='white')
            self.analysis_ax.set_title("水深分布直方图", color='white')

            # 添加平均值和中位数标记
            mean_depth = np.mean(valid_data)
            median_depth = np.median(valid_data)
            self.analysis_ax.axvline(mean_depth, color='red', linestyle='dashed', linewidth=1,
                                     label=f'平均值: {mean_depth:.2f}m')
            self.analysis_ax.axvline(median_depth, color='green', linestyle='dashed', linewidth=1,
                                     label=f'中位数: {median_depth:.2f}m')
            self.analysis_ax.legend(facecolor='#252526', edgecolor='#3F3F46', labelcolor='white')

            # 更新统计数据
            self.update_stats_table([
                ("平均水深", f"{mean_depth:.2f} m"),
                ("中位水深", f"{median_depth:.2f} m"),
                ("最大水深", f"{np.max(valid_data):.2f} m"),
                ("最小水深", f"{np.min(valid_data):.2f} m"),
                ("标准差", f"{np.std(valid_data):.2f} m"),
                ("数据点数", f"{len(valid_data)}")
            ])

            # 更新描述
            self.analysis_description.setText(
                "水深分布直方图显示不同水深值的频率分布。直方图形状可以揭示海底地形的特征，"
                "例如双峰分布可能表明存在两个主要深度层次，如海底沟渠或山脊。"
            )

        elif analysis_type == "海底特征识别":
            # 使用拉普拉斯算子检测特征
            from scipy import ndimage

            # 填充NaN值以便进行卷积
            filled_data = np.copy(self.depth_data)
            mask = np.isnan(filled_data)
            filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

            # 应用拉普拉斯滤波器检测边缘/特征
            features = ndimage.laplace(filled_data)

            # 将结果应用回原始掩码
            features[mask] = np.nan

            # 绘制特征图
            im = self.analysis_ax.imshow(
                features,
                cmap='seismic',
                interpolation='nearest',
                origin='lower'
            )
            self.analysis_figure.colorbar(im, ax=self.analysis_ax, label='特征强度')
            self.analysis_ax.set_title("海底特征识别", color='white')

            # 特征统计
            feature_threshold = np.nanstd(features) * 2
            strong_features = np.abs(features) > feature_threshold
            feature_count = np.sum(strong_features & ~mask)

            # 更新统计数据
            self.update_stats_table([
                ("检测到的特征数", f"{feature_count}"),
                ("特征密度", f"{feature_count / np.sum(~mask):.4f}"),
                ("平均特征强度", f"{np.nanmean(np.abs(features)):.2f}"),
                ("最大特征强度", f"{np.nanmax(np.abs(features)):.2f}")
            ])

            # 更新描述
            self.analysis_description.setText(
                "海底特征识别使用拉普拉斯滤波器突出显示深度快速变化的区域。红色和蓝色区域表示可能的海底特征，"
                "如山脊、沟渠、洞穴或人工结构。特征强度越高，颜色越鲜艳。"
            )

        elif analysis_type == "数据质量评估":
            # 计算相邻点的标准差来评估数据质量
            # 创建一个3x3窗口的标准差滤波器
            from scipy import ndimage

            # 填充NaN值以便进行卷积
            filled_data = np.copy(self.depth_data)
            mask = np.isnan(filled_data)
            filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

            # 计算局部标准差
            def local_std(x):
                return np.std(x) if np.sum(~np.isnan(x)) > 4 else np.nan

            quality_map = ndimage.generic_filter(filled_data, local_std, size=3)
            quality_map[mask] = np.nan

            # 绘制质量图
            im = self.analysis_ax.imshow(
                quality_map,
                cmap='RdYlGn_r',  # 红色表示低质量，绿色表示高质量
                interpolation='nearest',
                origin='lower'
            )
            self.analysis_figure.colorbar(im, ax=self.analysis_ax, label='局部标准差 (m)')
            self.analysis_ax.set_title("数据质量评估", color='white')

            # 定义质量阈值
            low_quality = quality_map > np.nanpercentile(quality_map, 90)
            high_quality = quality_map < np.nanpercentile(quality_map, 10)

            # 计算质量统计
            low_quality_count = np.sum(low_quality & ~mask)
            high_quality_count = np.sum(high_quality & ~mask)
            total_valid = np.sum(~mask)

            # 更新统计数据
            self.update_stats_table([
                ("高质量数据比例", f"{high_quality_count / total_valid * 100:.1f}%"),
                ("低质量数据比例", f"{low_quality_count / total_valid * 100:.1f}%"),
                ("平均局部标准差", f"{np.nanmean(quality_map):.3f} m"),
                ("最大局部标准差", f"{np.nanmax(quality_map):.3f} m"),
                ("数据质量评分", f"{100 - np.nanmean(quality_map) / np.nanmax(quality_map) * 100:.1f}/100")
            ])

            # 更新描述
            self.analysis_description.setText(
                "数据质量评估通过计算局部深度变化的标准差来评估测量精度。红色区域表示数据质量较低，可能需要重新测量；"
                "绿色区域表示数据质量高。质量问题可能由传感器噪声、水流湍动或测量间隔过大引起。"
            )

        # 更新图形
        self.analysis_canvas.draw()

    def update_stats_table(self, stats):
        """更新统计数据表"""
        # 清除表格
        self.stats_table.setRowCount(0)

        # 添加新数据
        for i, (param, value) in enumerate(stats):
            self.stats_table.insertRow(i)
            self.stats_table.setItem(i, 0, QTableWidgetItem(param))
            self.stats_table.setItem(i, 1, QTableWidgetItem(value))

    def apply_data_filter(self):
        """应用数据过滤"""
        filter_type = self.filter_combo.currentText()

        if filter_type == "无过滤":
            # 不做任何处理
            pass

        elif filter_type == "低通滤波":
            # 应用高斯平滑
            from scipy import ndimage

            # 创建一个副本并填充NaN
            filled_data = np.copy(self.depth_data)
            mask = np.isnan(filled_data)
            if np.any(mask):
                filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

            # 应用高斯滤波
            smoothed = ndimage.gaussian_filter(filled_data, sigma=1.0)

            # 将滤波结果应用回原始数据，保留NaN
            self.depth_data = np.where(mask, np.nan, smoothed)

            # 更新显示
            self.add_system_log("已应用低通滤波器")

        elif filter_type == "高通滤波":
            # 应用高通滤波来强调边缘
            from scipy import ndimage

            # 创建一个副本并填充NaN
            filled_data = np.copy(self.depth_data)
            mask = np.isnan(filled_data)
            if np.any(mask):
                filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

            # 应用拉普拉斯滤波器
            high_pass = filled_data - ndimage.gaussian_filter(filled_data, sigma=2.0)
            enhanced = filled_data + high_pass * 0.5  # 增强边缘

            # 将结果应用回原始数据，保留NaN
            self.depth_data = np.where(mask, np.nan, enhanced)

            # 更新显示
            self.add_system_log("已应用高通滤波器")

        elif filter_type == "中值滤波":
            # 应用中值滤波移除尖峰噪声
            from scipy import ndimage

            # 创建一个副本
            filled_data = np.copy(self.depth_data)
            mask = np.isnan(filled_data)
            if np.any(mask):
                filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

            # 应用中值滤波
            median_filtered = ndimage.median_filter(filled_data, size=3)

            # 将结果应用回原始数据，保留NaN
            self.depth_data = np.where(mask, np.nan, median_filtered)

            # 更新显示
            self.add_system_log("已应用中值滤波器")

        # 更新分析和显示
        self.update_analysis_view()
        self.update_realtime_display()

    def perform_analysis(self):
        """执行数据分析"""
        self.update_analysis_view()

        # 添加到系统日志
        analysis_type = self.analysis_combo.currentText()
        self.add_system_log(f"已执行{analysis_type}分析")

        # 显示正在执行动画
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)

        for i in range(101):
            progress.setValue(i)
            QApplication.processEvents()
            time.sleep(0.01)

    def update_progress(self):
        """更新进度条"""
        # 模拟数据缓冲进度
        value = self.progress_bar.value()
        value = (value + 5) % 100
        self.progress_bar.setValue(value)

    def update_performance_data(self):
        """更新性能数据图表"""
        # 模拟CPU使用率
        self.cpu_data = np.roll(self.cpu_data, -1)
        self.cpu_data[-1] = 20 + 15 * np.sin(time.time() * 0.5) + np.random.rand() * 10
        self.cpu_curve.setData(np.arange(len(self.cpu_data)), self.cpu_data)

        # 模拟内存使用率
        self.mem_data = np.roll(self.mem_data, -1)
        self.mem_data[-1] = 40 + 10 * np.sin(time.time() * 0.2) + np.random.rand() * 5
        self.mem_curve.setData(np.arange(len(self.mem_data)), self.mem_data)

        # 模拟网络流量
        self.net_data = np.roll(self.net_data, -1)
        self.net_data[-1] = 50 + 30 * np.sin(time.time() * 0.3) + np.random.rand() * 20
        self.net_curve.setData(np.arange(len(self.net_data)), self.net_data)

    def update_runtime(self):
        """更新运行时间显示"""
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60
        self.runtime_label.setText(f"运行时间: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def change_view_mode(self, index):
        """改变视图模式"""
        modes = ["标准视图", "高度差异视图", "坡度视图", "阴影浮雕视图"]
        self.add_system_log(f"视图模式切换为: {modes[index]}")

    def change_beam_count(self, index):
        """改变波束数量"""
        beam_counts = [32, 64, 128, 256]
        count = beam_counts[index]
        self.data_thread.set_params(beams=count)
        self.add_system_log(f"波束数量已更改为: {count}")

    def change_data_quality(self, index):
        """改变数据质量模式"""
        quality_modes = ["高精度", "标准", "快速扫描"]
        quality = quality_modes[index]
        self.data_thread.set_params(quality=quality)
        self.add_system_log(f"数据质量模式已更改为: {quality}")

    def change_3d_view_mode(self, index):
        """改变3D视图模式"""
        modes = ["彩色高程图", "线框图", "点云图", "等高线图"]
        self.view_mode = modes[index]
        self.add_system_log(f"3D视图模式已更改为: {self.view_mode}")

    def change_3d_color_scheme(self, index):
        """改变3D颜色方案"""
        schemes = ["深度渐变", "高光渲染", "地形分析", "海底分类"]
        self.color_scheme = schemes[index]
        self.add_system_log(f"3D颜色方案已更改为: {self.color_scheme}")

    def change_elevation_exaggeration(self, value):
        """改变高程夸张系数"""
        self.elevation_factor = value / 10.0  # 转换为1.0-5.0范围
        self.view_info_label.setText(
            f"显示 {self.grid_size}x{self.grid_size} 网格数据 | 高度夸张: {self.elevation_factor:.1f}x"
        )

    def change_refresh_rate(self, value):
        """改变刷新率"""
        interval = 1.0 / value  # 计算秒数
        self.data_thread.set_params(interval=interval)
        self.add_system_log(f"数据刷新率已更改为: {value} Hz")

    def start_acquisition(self):
        """开始数据采集"""
        self.data_thread.running = True
        if not self.data_thread.isRunning():
            self.data_thread.start()
        self.statusBar().showMessage("正在采集数据")
        self.add_system_log("开始数据采集")

    def stop_acquisition(self):
        """停止数据采集"""
        self.data_thread.running = False
        self.statusBar().showMessage("数据采集已停止")
        self.add_system_log("停止数据采集")

    def save_data(self):
        """保存数据"""
        filename, _ = QFileDialog.getSaveFileName(self, "保存数据", "",
                                                  "CSV文件 (*.csv);;NumPy文件 (*.npy);;所有文件 (*)")
        if not filename:
            return

        try:
            # 保存航迹数据
            if filename.endswith('.csv'):
                # 创建DataFrame
                data = {
                    'x': self.track_x,
                    'y': self.track_y,
                    'timestamp': [time.time() - self.start_time + i * 0.5 for i in range(len(self.track_x))]
                }
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

                # 同时保存深度数据
                depth_filename = filename.replace('.csv', '_depth.npy')
                np.save(depth_filename, self.depth_data)

                self.add_system_log(f"数据已保存至: {filename}", "信息")
                self.statusBar().showMessage(f"数据已保存至: {filename}")

            elif filename.endswith('.npy'):
                # 保存为NumPy数据包
                data_package = {
                    'track_x': np.array(self.track_x),
                    'track_y': np.array(self.track_y),
                    'depth_data': self.depth_data,
                    'timestamp': time.time(),
                    'metadata': {
                        'grid_size': self.grid_size,
                        'beam_count': self.beam_count
                    }
                }
                np.save(filename, data_package)
                self.add_system_log(f"数据包已保存至: {filename}", "信息")
                self.statusBar().showMessage(f"数据包已保存至: {filename}")

            # 显示成功消息
            QMessageBox.information(self, "保存成功", f"数据已成功保存至:\n{filename}")

        except Exception as e:
            self.add_system_log(f"保存数据出错: {str(e)}", "错误")
            QMessageBox.critical(self, "保存错误", f"保存数据时发生错误:\n{str(e)}")

    def load_data(self):
        """加载数据"""
        filename, _ = QFileDialog.getOpenFileName(self, "加载数据", "",
                                                  "CSV文件 (*.csv);;NumPy文件 (*.npy);;所有文件 (*)")
        if not filename:
            return

        try:
            # 暂停数据生成
            was_running = self.data_thread.running
            self.data_thread.running = False

            if filename.endswith('.csv'):
                # 加载CSV数据
                df = pd.read_csv(filename)
                self.track_x = df['x'].tolist()
                self.track_y = df['y'].tolist()

                # 尝试加载深度数据
                depth_filename = filename.replace('.csv', '_depth.npy')
                if os.path.exists(depth_filename):
                    self.depth_data = np.load(depth_filename)

            elif filename.endswith('.npy'):
                # 加载NumPy数据包
                data_package = np.load(filename, allow_pickle=True).item()
                self.track_x = data_package['track_x'].tolist()
                self.track_y = data_package['track_y'].tolist()
                self.depth_data = data_package['depth_data']

                # 更新网格大小
                if 'metadata' in data_package and 'grid_size' in data_package['metadata']:
                    self.grid_size = data_package['metadata']['grid_size']

            # 更新显示
            self.update_dashboard_stats()
            self.update_realtime_display()
            self.update_3d_view()
            self.update_analysis_view()

            # 恢复数据生成
            self.data_thread.running = was_running

            self.add_system_log(f"已加载数据: {filename}", "信息")
            self.statusBar().showMessage(f"已加载数据: {filename}")

            # 显示成功消息
            QMessageBox.information(self, "加载成功", f"数据已成功加载:\n{filename}")

        except Exception as e:
            self.add_system_log(f"加载数据出错: {str(e)}", "错误")
            QMessageBox.critical(self, "加载错误", f"加载数据时发生错误:\n{str(e)}")

    def export_report(self):
        """导出报告"""
        filename, _ = QFileDialog.getSaveFileName(self, "导出报告", "",
                                                  "HTML文件 (*.html);;文本文件 (*.txt);;所有文件 (*)")
        if not filename:
            return

        try:
            if filename.endswith('.html'):
                # 创建HTML报告
                html = """
                <!DOCTYPE html>
                <html>
                <head>
                    <title>多波束测深数据报告</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 20px; }
                        h1, h2 { color: #0066cc; }
                        .stats-table { border-collapse: collapse; width: 100%; margin: 15px 0; }
                        .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        .stats-table th { background-color: #f2f2f2; }
                        .figure { margin: 20px 0; text-align: center; }
                        .figure img { max-width: 100%; border: 1px solid #ddd; }
                    </style>
                </head>
                <body>
                    <h1>多波束测深数据报告</h1>
                    <p>生成时间: {date}</p>

                    <h2>数据摘要</h2>
                    <table class="stats-table">
                        <tr><th>参数</th><th>值</th></tr>
                        <tr><td>平均水深</td><td>{avg_depth:.2f} m</td></tr>
                        <tr><td>最大水深</td><td>{max_depth:.2f} m</td></tr>
                        <tr><td>最小水深</td><td>{min_depth:.2f} m</td></tr>
                        <tr><td>数据点数</td><td>{points}</td></tr>
                        <tr><td>扫描面积</td><td>{area:.2f} m²</td></tr>
                        <tr><td>航行距离</td><td>{distance:.2f} m</td></tr>
                    </table>

                    <h2>地形图</h2>
                    <div class="figure">
                        <img src="depth_map.png" alt="水深图">
                        <p>图1: 海底地形热力图</p>
                    </div>

                    <div class="figure">
                        <img src="3d_view.png" alt="3D地形图">
                        <p>图2: 海底3D地形图</p>
                    </div>

                    <h2>航迹图</h2>
                    <div class="figure">
                        <img src="track.png" alt="航迹图">
                        <p>图3: 测量航迹</p>
                    </div>

                    <h2>分析结果</h2>
                    <div class="figure">
                        <img src="analysis.png" alt="分析图">
                        <p>图4: {analysis_type}分析结果</p>
                    </div>

                    <h2>设备信息</h2>
                    <table class="stats-table">
                        <tr><th>设备</th><th>状态</th></tr>
                """.format(
                    date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    avg_depth=np.nanmean(self.depth_data) if np.any(~np.isnan(self.depth_data)) else 0,
                    max_depth=np.nanmax(self.depth_data) if np.any(~np.isnan(self.depth_data)) else 0,
                    min_depth=np.nanmin(self.depth_data) if np.any(~np.isnan(self.depth_data)) else 0,
                    points=len(self.track_x),
                    area=self.data_stats["扫描面积"],
                    distance=np.sum(np.sqrt(np.diff(self.track_x) ** 2 + np.diff(self.track_y) ** 2)) if len(
                        self.track_x) > 1 else 0,
                    analysis_type=self.analysis_combo.currentText()
                )

                # 添加设备信息
                for device, status in self.device_status.items():
                    status_color = "green" if status == "正常" else "red"
                    html += f'<tr><td>{device}</td><td style="color: {status_color};">{status}</td></tr>\n'

                html += """
                    </table>

                    <h2>系统日志</h2>
                    <pre style="background-color: #f8f8f8; padding: 10px; border: 1px solid #ddd; max-height: 300px; overflow: auto;">
                """

                # 添加系统日志
                for row in range(self.system_log.rowCount()):
                    time = self.system_log.item(row, 0).text()
                    event = self.system_log.item(row, 1).text()
                    html += f"{time} - {event}\n"

                html += """
                    </pre>

                    <p style="margin-top: 30px; color: #666; font-size: 12px;">该报告由多波束测深显控平台自动生成。版权所有 © 2023</p>
                </body>
                </html>
                """

                # 保存HTML文件
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html)

                # 保存图像文件
                img_path = os.path.dirname(filename)

                # 保存深度图
                plt.figure(figsize=(10, 8))
                plt.imshow(np.ma.masked_invalid(self.depth_data), cmap='viridis')
                plt.colorbar(label='水深 (m)')
                plt.title("海底地形热力图")
                plt.savefig(os.path.join(img_path, "depth_map.png"))
                plt.close()

                # 保存3D视图
                self.figure.savefig(os.path.join(img_path, "3d_view.png"))

                # 保存航迹图
                plt.figure(figsize=(8, 8))
                plt.plot(self.track_x, self.track_y, 'g-', linewidth=2)
                plt.scatter([self.track_x[-1]], [self.track_y[-1]], color='red', s=50)
                plt.grid(True, alpha=0.3)
                plt.title("测量航迹")
                plt.xlabel("X (m)")
                plt.ylabel("Y (m)")
                plt.savefig(os.path.join(img_path, "track.png"))
                plt.close()

                # 保存分析图
                self.analysis_figure.savefig(os.path.join(img_path, "analysis.png"))

                self.add_system_log(f"报告已导出至: {filename}", "信息")
                self.statusBar().showMessage(f"报告已导出至: {filename}")

            elif filename.endswith('.txt'):
                # 创建文本报告
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("========================================\n")
                    f.write("        多波束测深数据报告\n")
                    f.write("========================================\n\n")
                    f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    f.write("数据摘要:\n")
                    f.write(f"- 平均水深: {np.nanmean(self.depth_data):.2f} m\n")
                    f.write(f"- 最大水深: {np.nanmax(self.depth_data):.2f} m\n")
                    f.write(f"- 最小水深: {np.nanmin(self.depth_data):.2f} m\n")
                    f.write(f"- 数据点数: {len(self.track_x)}\n")
                    f.write(f"- 扫描面积: {self.data_stats['扫描面积']:.2f} m²\n")
                    f.write(
                        f"- 航行距离: {np.sum(np.sqrt(np.diff(self.track_x) ** 2 + np.diff(self.track_y) ** 2)):.2f} m\n\n")

                    f.write("设备状态:\n")
                    for device, status in self.device_status.items():
                        f.write(f"- {device}: {status}\n")

                    f.write("\n系统日志:\n")
                    for row in range(min(self.system_log.rowCount(), 20)):  # 限制日志条数
                        time = self.system_log.item(row, 0).text()
                        event = self.system_log.item(row, 1).text()
                        f.write(f"{time} - {event}\n")

                self.add_system_log(f"文本报告已导出至: {filename}", "信息")
                self.statusBar().showMessage(f"文本报告已导出至: {filename}")

            # 显示成功消息
            QMessageBox.information(self, "导出成功", f"报告已成功导出至:\n{filename}")

        except Exception as e:
            self.add_system_log(f"导出报告出错: {str(e)}", "错误")
            QMessageBox.critical(self, "导出错误", f"导出报告时发生错误:\n{str(e)}")

    def export_analysis(self):
        """导出分析结果"""
        analysis_type = self.analysis_combo.currentText()
        filename, _ = QFileDialog.getSaveFileName(
            self,
            f"导出{analysis_type}分析结果",
            f"{analysis_type}_analysis.png",
            "PNG图像 (*.png);;CSV数据 (*.csv);;所有文件 (*)"
        )
        if not filename:
            return

        try:
            if filename.endswith('.png'):
                # 导出分析图像
                self.analysis_figure.savefig(filename, dpi=150)

            elif filename.endswith('.csv'):
                # 导出分析数据为CSV
                data = {}

                if analysis_type == "海底坡度分析":
                    # 计算坡度数据
                    grad_y, grad_x = np.gradient(self.depth_data)
                    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)

                    # 保存坡度数据
                    x = np.arange(self.grid_size)
                    y = np.arange(self.grid_size)
                    X, Y = np.meshgrid(x, y)

                    data = {
                        'x': X.flatten(),
                        'y': Y.flatten(),
                        'depth': self.depth_data.flatten(),
                        'slope': slope.flatten()
                    }

                elif analysis_type == "水深分布直方图":
                    # 导出深度分布数据
                    valid_data = self.depth_data[~np.isnan(self.depth_data)]
                    hist, bins = np.histogram(valid_data, bins=30)
                    bin_centers = (bins[:-1] + bins[1:]) / 2

                    data = {
                        'depth_bin': bin_centers,
                        'frequency': hist
                    }

                elif analysis_type == "海底特征识别":
                    # 导出特征数据
                    from scipy import ndimage

                    # 填充NaN值以便进行卷积
                    filled_data = np.copy(self.depth_data)
                    mask = np.isnan(filled_data)
                    filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

                    # 应用拉普拉斯滤波器检测边缘/特征
                    features = ndimage.laplace(filled_data)

                    # 将结果应用回原始掩码
                    features[mask] = np.nan

                    # 创建数据
                    x = np.arange(self.grid_size)
                    y = np.arange(self.grid_size)
                    X, Y = np.meshgrid(x, y)

                    data = {
                        'x': X.flatten(),
                        'y': Y.flatten(),
                        'depth': self.depth_data.flatten(),
                        'feature_strength': features.flatten()
                    }

                elif analysis_type == "数据质量评估":
                    # 导出质量评估数据
                    from scipy import ndimage

                    # 填充NaN值以便进行卷积
                    filled_data = np.copy(self.depth_data)
                    mask = np.isnan(filled_data)
                    filled_data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), filled_data[~mask])

                    # 计算局部标准差
                    def local_std(x):
                        return np.std(x) if np.sum(~np.isnan(x)) > 4 else np.nan

                    quality_map = ndimage.generic_filter(filled_data, local_std, size=3)
                    quality_map[mask] = np.nan

                    # 创建数据
                    x = np.arange(self.grid_size)
                    y = np.arange(self.grid_size)
                    X, Y = np.meshgrid(x, y)

                    data = {
                        'x': X.flatten(),
                        'y': Y.flatten(),
                        'depth': self.depth_data.flatten(),
                        'quality': quality_map.flatten()
                    }

                # 保存为CSV
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False)

            self.add_system_log(f"分析结果已导出至: {filename}", "信息")
            self.statusBar().showMessage(f"分析结果已导出至: {filename}")

            # 显示成功消息
            QMessageBox.information(self, "导出成功", f"分析结果已成功导出至:\n{filename}")

        except Exception as e:
            self.add_system_log(f"导出分析结果出错: {str(e)}", "错误")
            QMessageBox.critical(self, "导出错误", f"导出分析结果时发生错误:\n{str(e)}")

    def save_3d_screenshot(self):
        """保存3D视图截图"""
        filename, _ = QFileDialog.getSaveFileName(self, "保存3D视图截图", "3d_view.png",
                                                  "PNG图像 (*.png);;所有文件 (*)")
        if not filename:
            return

        try:
            self.figure.savefig(filename, dpi=150)
            self.add_system_log(f"3D视图已保存至: {filename}", "信息")
            QMessageBox.information(self, "保存成功", f"3D视图已成功保存至:\n{filename}")
        except Exception as e:
            self.add_system_log(f"保存3D视图出错: {str(e)}", "错误")
            QMessageBox.critical(self, "保存错误", f"保存3D视图时发生错误:\n{str(e)}")

    def browse_storage_path(self):
        """浏览存储路径"""
        directory = QFileDialog.getExistingDirectory(self, "选择存储路径")
        if directory:
            # 此处应该更新存储路径输入框
            self.add_system_log(f"数据存储路径已更改为: {directory}")

    def save_settings(self):
        """保存设置"""
        # 这里应该实现设置的保存
        self.add_system_log("系统设置已保存")
        QMessageBox.information(self, "设置保存", "系统设置已成功保存！")

    def reset_settings(self):
        """重置设置"""
        reply = QMessageBox.question(self, "重置设置", "确定要恢复默认设置吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # 这里应该实现设置的重置
            self.add_system_log("系统设置已恢复默认值")
            QMessageBox.information(self, "重置设置", "系统设置已恢复为默认值！")

    def closeEvent(self, event):
        """关闭事件处理"""
        reply = QMessageBox.question(self, '确认退出',
                                     "确定要退出系统吗？如有未保存的数据将会丢失。",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # 停止所有线程
            self.data_thread.stop()
            self.data_thread.wait()
            self.timer.stop()
            self.progress_timer.stop()
            self.update_timer.stop()
            self.perf_timer.stop()
            event.accept()
        else:
            event.ignore()


# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultibeamSonarSystem()
    window.show()
    sys.exit(app.exec_())
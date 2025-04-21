import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from image_processor import ImageProcessor
import cv2
import qdarkstyle
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvas

# 中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ModernGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = ImageProcessor()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('路面缺陷智能检测系统')
        self.setGeometry(100, 100, 1600, 1000)
        
        # 创建主窗口部件
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 创建主布局
        layout = QHBoxLayout(main_widget)
        layout.setSpacing(10)  # 增加组件间距
        
        # 左侧控制面板
        control_panel = self.create_control_panel()
        control_panel.setMinimumWidth(400)  # 设置最小宽度
        control_panel.setMaximumWidth(500)  # 设置最大宽度
        
        # 右侧图像显示区域
        display_panel = self.create_display_panel()
        
        # 添加分割线
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(control_panel)
        splitter.addWidget(display_panel)
        splitter.setSizes([400, 1200])  # 调整左右面板的比例
        
        layout.addWidget(splitter)
        
        # 设置状态栏
        self.statusBar().showMessage('就绪')
        
        # 应用深色主题
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    def create_control_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)
        
        # 恢复原来的样式
        style = """
            QMainWindow {
                background-color: #f5f6fa;
            }
            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 8px;
                margin-top: 1.5ex;
                padding: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #2980b9;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 5px;
                padding: 8px;
                min-height: 25px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #2472a4;
            }
            QLabel {
                color: #34495e;
                font-size: 12px;
                font-weight: bold;
                margin: 2px;
            }
            QSlider::handle:horizontal {
                background: #3498db;
                border-radius: 7px;
                width: 14px;
                margin: -4px 0;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bdc3c7;
                height: 8px;
                background: #ecf0f1;
                margin: 2px 0;
                border-radius: 4px;
            }
            QTextEdit {
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                padding: 5px;
                background-color: white;
                font-size: 12px;
            }
        """
        self.setStyleSheet(style)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # 创建内容容器
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(15)
        
        # 1. 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()
        file_layout.setSpacing(8)  # 增加按钮间距
        
        self.load_btn = QPushButton("📂 加载单张图片")
        self.save_btn = QPushButton("💾 保存结果")
        
        # 添加批处理按钮和选择框到文件操作组
        batch_group = QGroupBox("批处理设置")
        batch_group.setStyleSheet("""
            QGroupBox {
                border: 2px solid #3498db;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: #3498db;
            }
        """)
        batch_layout = QGridLayout()
        batch_layout.setSpacing(10)
        
        # 添加处理方法选择
        method_label = QLabel("处理方法:")
        method_label.setStyleSheet("color: #2c3e50;")
        self.process_method = QComboBox()
        self.process_method.addItems(["传统方法", "AI方法"])
        self.process_method.setStyleSheet("""
            QComboBox {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                padding: 3px;
                min-width: 100px;
                background: white;
            }
            QComboBox:hover {
                border-color: #3498db;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
        """)
        
        # 添加AI检测模式选择
        ai_mode_label = QLabel("AI检测模式:")
        ai_mode_label.setStyleSheet("color: #2c3e50;")
        self.ai_mode = QComboBox()
        self.ai_mode.addItems(["边界框检测", "分割检测", "混合检测"])
        self.ai_mode.setEnabled(False)  # 初始禁用
        self.ai_mode.setStyleSheet(self.process_method.styleSheet())
        
        # 添加批处理按钮
        batch_btn = QPushButton("📦 批量处理")
        batch_btn.setStyleSheet("""
            QPushButton {
                background-color: #9b59b6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 15px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #8e44ad;
            }
            QPushButton:pressed {
                background-color: #6c3483;
            }
        """)
        
        # 使用网格布局排列组件
        batch_layout.addWidget(method_label, 0, 0)
        batch_layout.addWidget(self.process_method, 0, 1)
        batch_layout.addWidget(ai_mode_label, 1, 0)
        batch_layout.addWidget(self.ai_mode, 1, 1)
        batch_layout.addWidget(batch_btn, 2, 0, 1, 2, Qt.AlignCenter)
        
        # 设置列拉伸
        batch_layout.setColumnStretch(1, 1)
        
        batch_group.setLayout(batch_layout)
        file_layout.addWidget(batch_group)
        
        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(QLabel("批处理方法:"))
        file_layout.addWidget(self.process_method)
        file_layout.addWidget(batch_btn)
        file_layout.addWidget(self.save_btn)
        
        file_group.setLayout(file_layout)
        
        # 连接文件操作信号（只在这里连接一次）
        self.load_btn.clicked.connect(self.load_image)
        self.save_btn.clicked.connect(self.save_result)
        # batch_process_btn.clicked.connect(self.batch_process)
        
        # 连接信号
        self.process_method.currentTextChanged.connect(self.on_process_method_changed)
        # batch_btn.clicked.connect(self.batch_process)
        
        # 2. 图像分析组
        analysis_group = QGroupBox("图像分析")
        analysis_layout = QVBoxLayout()
        analysis_layout.setSpacing(8)
        
        # 调整直方图显示区域
        self.hist_label = QLabel()
        self.hist_label.setMinimumHeight(200)
        self.hist_label.setMaximumHeight(250)
        
        # 调整统计信息显示区域
        self.hist_info = QTextEdit()
        self.hist_info.setMinimumHeight(80)
        self.hist_info.setMaximumHeight(100)
        
        analysis_layout.addWidget(self.hist_label)
        analysis_layout.addWidget(self.hist_info)
        
        analysis_group.setLayout(analysis_layout)
        
        # 3. 图像处理组
        process_group = QGroupBox("图像处理")
        process_layout = QVBoxLayout()
        process_layout.setSpacing(10)
        
        # 3.1 基础调节 - 使用网格布局
        basic_adjust = QGroupBox("基础调节")
        basic_layout = QGridLayout()
        basic_layout.setVerticalSpacing(8)
        basic_layout.setHorizontalSpacing(10)
        
        # 亮度对比度
        brightness_label = QLabel("亮度:")
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_value = QLabel("0")
        
        contrast_label = QLabel("对比度:")
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 300)
        self.contrast_value = QLabel("100")
        
        basic_layout.addWidget(brightness_label, 0, 0)
        basic_layout.addWidget(self.brightness_slider, 0, 1)
        basic_layout.addWidget(self.brightness_value, 0, 2)
        basic_layout.addWidget(contrast_label, 1, 0)
        basic_layout.addWidget(self.contrast_slider, 1, 1)
        basic_layout.addWidget(self.contrast_value, 1, 2)
        
        basic_adjust.setLayout(basic_layout)
        process_layout.addWidget(basic_adjust)
        
        # 3.2 高级参数 - 使用网格布局
        advanced_params = QGroupBox("高级参数")
        params_layout = QGridLayout()
        params_layout.setVerticalSpacing(8)
        params_layout.setHorizontalSpacing(10)
        
        # Canny参数
        canny_low_label = QLabel("Canny低阈值:")
        self.canny_low_slider = QSlider(Qt.Horizontal)
        self.canny_low_slider.setRange(0, 255)
        self.canny_low_value = QLabel("50")
        
        canny_high_label = QLabel("Canny高阈值:")
        self.canny_high_slider = QSlider(Qt.Horizontal)
        self.canny_high_slider.setRange(0, 255)
        self.canny_high_value = QLabel("150")
        
        # FFT参数
        fft_radius_label = QLabel("FFT滤波半径:")
        self.fft_radius_slider = QSlider(Qt.Horizontal)
        self.fft_radius_slider.setRange(10, 100)
        self.fft_radius_value = QLabel("30")
        
        # 形态学参数
        morph_size_label = QLabel("形态学核大小:")
        self.morph_size_slider = QSlider(Qt.Horizontal)
        self.morph_size_slider.setRange(3, 21)
        self.morph_size_value = QLabel("3")
        
        # 添加边缘连接控制组件
        edge_connect_group = QGroupBox("边缘连接")
        edge_connect_layout = QGridLayout()
        edge_connect_layout.setVerticalSpacing(8)
        edge_connect_layout.setHorizontalSpacing(10)
        
        # 启用复选框
        self.edge_connect_checkbox = QCheckBox("启用边缘连接")
        self.edge_connect_checkbox.setChecked(False)
        edge_connect_layout.addWidget(self.edge_connect_checkbox, 0, 0, 1, 3)
        
        # 最小阈值
        min_threshold_label = QLabel("最小连接阈值:")
        self.min_threshold_slider = QSlider(Qt.Horizontal)
        self.min_threshold_slider.setRange(1, 50)
        self.min_threshold_slider.setValue(5)
        self.min_threshold_slider.setEnabled(False)
        self.min_threshold_value = QLabel("5")
        edge_connect_layout.addWidget(min_threshold_label, 1, 0)
        edge_connect_layout.addWidget(self.min_threshold_slider, 1, 1)
        edge_connect_layout.addWidget(self.min_threshold_value, 1, 2)
        
        # 最大阈值
        max_threshold_label = QLabel("最大连接阈值:")
        self.max_threshold_slider = QSlider(Qt.Horizontal)
        self.max_threshold_slider.setRange(5, 100)
        self.max_threshold_slider.setValue(15)
        self.max_threshold_slider.setEnabled(False)
        self.max_threshold_value = QLabel("15")
        edge_connect_layout.addWidget(max_threshold_label, 2, 0)
        edge_connect_layout.addWidget(self.max_threshold_slider, 2, 1)
        edge_connect_layout.addWidget(self.max_threshold_value, 2, 2)
        
        edge_connect_group.setLayout(edge_connect_layout)
        
        # 连接信号
        self.edge_connect_checkbox.stateChanged.connect(self.on_edge_connect_changed)
        self.min_threshold_slider.valueChanged.connect(self.update_min_threshold)
        self.max_threshold_slider.valueChanged.connect(self.update_max_threshold)
        
        params_layout.addWidget(canny_low_label, 0, 0)
        params_layout.addWidget(self.canny_low_slider, 0, 1)
        params_layout.addWidget(self.canny_low_value, 0, 2)
        params_layout.addWidget(canny_high_label, 1, 0)
        params_layout.addWidget(self.canny_high_slider, 1, 1)
        params_layout.addWidget(self.canny_high_value, 1, 2)
        params_layout.addWidget(fft_radius_label, 2, 0)
        params_layout.addWidget(self.fft_radius_slider, 2, 1)
        params_layout.addWidget(self.fft_radius_value, 2, 2)
        params_layout.addWidget(morph_size_label, 3, 0)
        params_layout.addWidget(self.morph_size_slider, 3, 1)
        params_layout.addWidget(self.morph_size_value, 3, 2)
        params_layout.addWidget(edge_connect_group, 4, 0, 1, 3)
        
        advanced_params.setLayout(params_layout)
        process_layout.addWidget(advanced_params)
        
        # 3.3 处理按钮组 - 使用网格布局
        buttons_group = QGroupBox("处理操作")
        buttons_layout = QGridLayout()
        buttons_layout.setVerticalSpacing(8)
        buttons_layout.setHorizontalSpacing(10)
        
        # 添加直方图相关按钮
        hist_eq_btn = QPushButton("📊 直方图均衡化")
        clahe_btn = QPushButton("📈 自适应直方图均衡化")
        hist_info_btn = QPushButton("📋 详细信息")
        
        hist_eq_btn.clicked.connect(self.apply_histogram_equalization)
        clahe_btn.clicked.connect(self.apply_clahe)
        hist_info_btn.clicked.connect(self.show_detailed_info)
        
        buttons_layout.addWidget(hist_eq_btn, 0, 0)
        buttons_layout.addWidget(clahe_btn, 0, 1)
        buttons_layout.addWidget(hist_info_btn, 0, 2)
        
        # 基础处理按钮
        enhance_btn = QPushButton("🔆 图像增强")
        edge_btn = QPushButton("📐 边缘检测")
        fft_btn = QPushButton("🌊 FFT滤波")
        
        # 形态学处理按钮
        erode_btn = QPushButton("⚪ 腐蚀")
        dilate_btn = QPushButton("⭕ 膨胀")
        open_btn = QPushButton("📂 开运算")
        close_btn = QPushButton("📁 闭运算")
        gradient_btn = QPushButton("📊 形态学梯度")
        
        buttons_layout.addWidget(enhance_btn, 1, 0)
        buttons_layout.addWidget(edge_btn, 1, 1)
        buttons_layout.addWidget(fft_btn, 1, 2)
        buttons_layout.addWidget(erode_btn, 2, 0)
        buttons_layout.addWidget(dilate_btn, 2, 1)
        buttons_layout.addWidget(open_btn, 2, 2)
        buttons_layout.addWidget(close_btn, 3, 0)
        buttons_layout.addWidget(gradient_btn, 3, 1, 1, 2)
        
        buttons_group.setLayout(buttons_layout)
        process_layout.addWidget(buttons_group)
        
        process_group.setLayout(process_layout)
        
        # 4. 缺陷检测组
        detect_group = QGroupBox("缺陷检测")
        detect_layout = QVBoxLayout()
        detect_layout.setSpacing(8)
        
        # 添加检测模式选择
        mode_group = QGroupBox("检测模式")
        mode_layout = QHBoxLayout()
        
        self.bbox_radio = QRadioButton("边界框")
        self.segment_radio = QRadioButton("分割")
        self.both_radio = QRadioButton("混合")
        self.bbox_radio.setChecked(True)  # 默认选择边界框模式
        
        mode_layout.addWidget(self.bbox_radio)
        mode_layout.addWidget(self.segment_radio)
        mode_layout.addWidget(self.both_radio)
        mode_group.setLayout(mode_layout)
        detect_layout.addWidget(mode_group)
        
        # 检测按钮
        detect_all_btn = QPushButton("🔍 全部缺陷检测")
        detect_all_btn.setStyleSheet("background-color: #27ae60;")
        detect_crack_btn = QPushButton("↔️ 裂缝检测")
        detect_pothole_btn = QPushButton("⭕ 坑洼检测")
        detect_water_btn = QPushButton("💧 积水检测")
        
        # 添加AI检测按钮
        detect_ai_btn = QPushButton("🤖 AI智能检测")
        detect_ai_btn.setStyleSheet("background-color: #3498db;")
        # 移除这里的信号连接
        # detect_ai_btn.clicked.connect(self.detect_defects_ai)
        
        detect_layout.addWidget(detect_all_btn)
        detect_layout.addWidget(detect_crack_btn)
        detect_layout.addWidget(detect_pothole_btn)
        detect_layout.addWidget(detect_water_btn)
        detect_layout.addWidget(detect_ai_btn)
        
        # 检测结果显示
        self.result_text = QTextEdit()
        self.result_text.setMinimumHeight(80)
        self.result_text.setMaximumHeight(100)
        self.result_text.setReadOnly(True)
        detect_layout.addWidget(self.result_text)
        
        detect_group.setLayout(detect_layout)
        
        # 连接批处理按钮信号
        batch_btn.clicked.connect(self.batch_process)
        
        # 5. 重置按钮
        reset_btn = QPushButton("🔄 重置图像")
        reset_btn.setStyleSheet("background-color: #e74c3c;")
        
        # 在添加组件到content_layout之前，先连接所有信号
        # 连接滑块信号
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        self.canny_low_slider.valueChanged.connect(self.update_canny_low)
        self.canny_high_slider.valueChanged.connect(self.update_canny_high)
        self.fft_radius_slider.valueChanged.connect(self.update_fft_radius)
        self.morph_size_slider.valueChanged.connect(self.update_morph_size)
        
        # 连接图像处理按钮信号
        enhance_btn.clicked.connect(self.enhance_image)
        edge_btn.clicked.connect(self.detect_edges)
        fft_btn.clicked.connect(self.apply_fft)
        
        # 连接形态学处理按钮信号
        erode_btn.clicked.connect(lambda: self.apply_morph_op('erode'))
        dilate_btn.clicked.connect(lambda: self.apply_morph_op('dilate'))
        open_btn.clicked.connect(lambda: self.apply_morph_op('open'))
        close_btn.clicked.connect(lambda: self.apply_morph_op('close'))
        gradient_btn.clicked.connect(lambda: self.apply_morph_op('gradient'))
        
        # 连接检测按钮信号
        detect_all_btn.clicked.connect(self.detect_defects)
        detect_crack_btn.clicked.connect(self.detect_cracks_only)
        detect_pothole_btn.clicked.connect(self.detect_potholes_only)
        detect_water_btn.clicked.connect(self.detect_water_only)
        detect_ai_btn.clicked.connect(self.detect_defects_ai)
        
        # 连接重置按钮信号
        reset_btn.clicked.connect(self.reset_image)
        
        # 添加所有组件到内容布局
        content_layout.addWidget(file_group)
        content_layout.addWidget(analysis_group)
        content_layout.addWidget(process_group)
        content_layout.addWidget(detect_group)
        content_layout.addWidget(reset_btn)
        content_layout.addStretch()
        
        # 将内容容器添加到滚动区域
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        return panel

    def create_display_panel(self):
        """创建右侧显示面板，包含所有图像处理结果"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 创建网格布局容器
        results_widget = QWidget()
        self.results_layout = QGridLayout(results_widget)
        self.results_layout.setSpacing(15)  # 增加网格间距
        
        # 创建处理结果显示区域字典
        self.result_widgets = {
            'input': self.create_result_widget("原始图像", 0, 0, 1, 2),
            'histogram_eq': self.create_result_widget("直方图均衡化", 1, 0),
            'clahe': self.create_result_widget("自适应直方图均衡化", 1, 1),
            'enhance': self.create_result_widget("图像增强", 2, 0),
            'edge': self.create_result_widget("边缘检测", 2, 1),
            'fft': self.create_result_widget("FFT滤波", 3, 0),
            'morph_erode': self.create_result_widget("腐蚀", 3, 1),
            'morph_dilate': self.create_result_widget("膨胀", 4, 0),
            'morph_open': self.create_result_widget("开运算", 4, 1),
            'morph_close': self.create_result_widget("闭运算", 5, 0),
            'morph_gradient': self.create_result_widget("形态学梯度", 5, 1),
            'defect': self.create_result_widget("缺陷检测", 6, 0, 1, 2)
        }
        
        # 设置网格的列宽比例
        self.results_layout.setColumnStretch(0, 1)
        self.results_layout.setColumnStretch(1, 1)
        
        # 添加滚动区域
        scroll = QScrollArea()
        scroll.setWidget(results_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(1000)  # 设置最小宽度
        layout.addWidget(scroll)
        
        return panel

    def create_result_widget(self, title, row, col, rowspan=1, colspan=1):
        """创建处理结果显示组件"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 标题和工具栏
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("""
            font-weight: bold;
            font-size: 13px;
            color: #2c3e50;
        """)
        header_layout.addWidget(title_label)
        
        # 添加复制按钮
        if title != "原始图像":
            copy_btn = QPushButton("📋")
            copy_btn.setToolTip("复制图像")
            copy_btn.setMaximumWidth(30)
            copy_btn.clicked.connect(self.copy_selected_image)
            header_layout.addWidget(copy_btn)
        
        # 创建选择状态标签
        selected_label = QLabel("⚪")
        selected_label.setObjectName("selected_label")
        if title == "原始图像":  # 原图不显示选择状态
            selected_label.hide()
        header_layout.addWidget(selected_label, alignment=Qt.AlignRight)
        
        layout.addWidget(header)
        
        # 图像显示
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setMinimumSize(450, 350)
        
        # 初始状态样式
        if title == "原始图像":
            image_label.setStyleSheet("""
                QLabel {
                    border: 2px solid #3498db;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                }
            """)
            image_label.setMinimumSize(900, 400)
        else:
            # 未处理状态的样式
            image_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #bdc3c7;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                    color: #95a5a6;
                }
            """)
            image_label.setText("等待处理")
        
        # 添加双击事件
        image_label.mouseDoubleClickEvent = lambda e: self.show_image_viewer(image_label)
        
        # 添加点击事件
        if title != "原始图像":
            image_label.mousePressEvent = lambda e: self.select_result_image(image_label)
        
        layout.addWidget(image_label)
        
        # 添加到网格布局
        self.results_layout.addWidget(widget, row, col, rowspan, colspan)
        
        return {
            'widget': widget,
            'label': image_label,
            'selected': False,
            'has_result': False,
            'selected_label': selected_label
        }

    def select_result_image(self, clicked_label):
        """选择处理结果图像作为下一步处理的输入"""
        # 查找对应的结果组件
        clicked_result = None
        for key, result in self.result_widgets.items():
            if result['label'] == clicked_label:
                clicked_result = result
                break
        
        # 如果没有找到对应组件或是原图，直接返回
        if not clicked_result or clicked_result == self.result_widgets['input']:
            return
        
        # 只有存在处理结果时才能选择
        if clicked_result['has_result']:
            # 取消所有其他选择
            for result in self.result_widgets.values():
                if result != clicked_result:
                    result['selected'] = False
                    result['widget'].setStyleSheet("")
                    result['selected_label'].setText("⚪")
            
            # 切换当前选择状态
            clicked_result['selected'] = not clicked_result['selected']
            if clicked_result['selected']:
                clicked_result['widget'].setStyleSheet("border: 3px solid #27ae60;")
                clicked_result['selected_label'].setText("🔵")
                # 获取图像数据并设置为当前处理图像
                pixmap = clicked_label.pixmap()
                if pixmap:
                    image = self.pixmap_to_cv2(pixmap)
                    self.processor.current_image = image
                    # 更新图像信息显示
                    self.update_clicked_image_info(image, clicked_result['widget'].findChild(QLabel).text())
            else:
                clicked_result['widget'].setStyleSheet("")
                clicked_result['selected_label'].setText("⚪")
                # 如果取消选择，恢复为原图
                self.processor.current_image = self.processor.original_image.copy()
                # 更新为原图信息
                self.update_clicked_image_info(self.processor.original_image, "原始图像")

    def update_clicked_image_info(self, cv_image, source_name):
        """更新点击图像的信息显示"""
        if cv_image is None:
            return
        
        # 创建直方图
        plt.figure(figsize=(5, 4))
        plt.title(f"当前选中: {source_name}", pad=10, fontsize=10)
        
        # 绘制三通道直方图
        colors = ('蓝色', '绿色', '红色')
        for i, (color, name) in enumerate(zip(('b', 'g', 'r'), colors)):
            hist = cv2.calcHist([cv_image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=name)
        
        plt.legend(loc='upper right', fontsize=8)
        plt.xlabel('像素值', fontsize=9)
        plt.ylabel('频率', fontsize=9)
        plt.xlim([0, 256])
        
        # 将matplotlib图像转换为QPixmap
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        hist_image = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
        plt.close()
        
        # 显示直方图
        pixmap = QPixmap.fromImage(hist_image)
        self.hist_label.setPixmap(pixmap.scaled(
            self.hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 计算并显示统计信息
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        median_val = np.median(gray)
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        # 计算梯度信息
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_mag)
        
        info_text = f"<b>当前选中: {source_name}</b><br><br>"
        info_text += f"<table style='font-size: 12px;'>"
        info_text += f"<tr><td>图像尺寸:</td><td>{cv_image.shape[1]}×{cv_image.shape[0]}</td></tr>"
        info_text += f"<tr><td>均值:</td><td>{mean_val:.2f}</td></tr>"
        info_text += f"<tr><td>标准差:</td><td>{std_val:.2f}</td></tr>"
        info_text += f"<tr><td>中值:</td><td>{median_val:.2f}</td></tr>"
        info_text += f"<tr><td>最小值:</td><td>{min_val}</td></tr>"
        info_text += f"<tr><td>最大值:</td><td>{max_val}</td></tr>"
        info_text += f"<tr><td>平均梯度:</td><td>{mean_gradient:.2f}</td></tr>"
        info_text += "</table>"
        
        self.hist_info.setHtml(info_text)

    def get_current_source_image(self):
        """获取当前处理的源图像"""
        # 检查是否有选中的处理结果
        for result in self.result_widgets.values():
            if result['selected'] and result['has_result']:
                pixmap = result['label'].pixmap()
                if pixmap:
                    return self.pixmap_to_cv2(pixmap)
        
        # 如果没有选中的结果，返回原图副本
        return self.processor.original_image.copy()

    def pixmap_to_cv2(self, pixmap):
        """将QPixmap转换为OpenCV图像格式"""
        qimage = pixmap.toImage()
        width = qimage.width()
        height = qimage.height()
        ptr = qimage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    def update_result_display(self, image, operation_type):
        """更新指定操作类型的结果显示"""
        if operation_type in self.result_widgets:
            result_widget = self.result_widgets[operation_type]
            # 如果是原图，不允许更改
            if operation_type == 'input':
                return
            
            height, width = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            q_image = QImage(image_rgb.data, width, height, width * 3, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(
                result_widget['label'].size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            result_widget['label'].setPixmap(scaled_pixmap)
            result_widget['has_result'] = True
            result_widget['label'].setText("")  # 清除等待处理文字
            result_widget['label'].setStyleSheet("""
                QLabel {
                    border: 2px solid #bdc3c7;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                }
            """)

    def load_image(self):
        """加载图片并显示"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)"
            )
            if not file_path:  # 如果用户取消选择，直接返回
                return
                
            # 加载图片
            self.processor.load_image(file_path)
            
            # 显示原图（特殊处理）
            if 'input' in self.result_widgets:
                input_widget = self.result_widgets['input']
                height, width = self.processor.original_image.shape[:2]
                image_rgb = cv2.cvtColor(self.processor.original_image, cv2.COLOR_BGR2RGB)
                q_image = QImage(image_rgb.data, width, height, width * 3, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(
                    input_widget['label'].size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                input_widget['label'].setPixmap(scaled_pixmap)
            
            # 重置其他所有显示区域
            self.reset_result_displays()
            
            # 更新直方图
            self.update_histogram()
            self.statusBar().showMessage('图片加载成功')
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图片失败：{str(e)}")

    def reset_result_displays(self):
        """重置所有结果显示区域（除了原图）"""
        for key, result in self.result_widgets.items():
            if key != 'input':
                result['label'].clear()
                result['label'].setText("等待处理")
                result['selected'] = False
                result['has_result'] = False
                result['label'].setStyleSheet("""
                    QLabel {
                        border: 2px dashed #bdc3c7;
                        border-radius: 5px;
                        background-color: #f8f9fa;
                        color: #95a5a6;
                    }
                """)
                result['selected_label'].setText("⚪")

    def reset_image(self):
        """重置图像处理状态"""
        if self.processor.original_image is None:
            return
        
        # 检查是否有选中的结果
        selected_result = None
        for key, result in self.result_widgets.items():
            if result['selected'] and key != 'input':
                selected_result = result
                break
        
        if selected_result:
            # 只重置选中的结果
            selected_result['label'].clear()
            selected_result['label'].setText("等待处理")
            selected_result['selected'] = False
            selected_result['has_result'] = False
            selected_result['label'].setStyleSheet("""
                QLabel {
                    border: 2px dashed #bdc3c7;
                    border-radius: 5px;
                    background-color: #f8f9fa;
                    color: #95a5a6;
                }
            """)
            selected_result['selected_label'].setText("⚪")
        else:
            # 重置所有结果（除原图外）
            self.reset_result_displays()
        
        # 重置处理器状态
        self.processor.current_image = self.processor.original_image.copy()
        self.update_histogram()  # 更新直方图显示
        self.statusBar().showMessage('图像已重置')

    def display_comparison(self, left_image, right_image):
        """左右对比显示两张图片"""
        if left_image is None or right_image is None:
            return
        
        # 确保两张图片尺寸相同
        h1, w1 = left_image.shape[:2]
        h2, w2 = right_image.shape[:2]
        h = max(h1, h2)
        w = max(w1, w2)
        
        # 调整图片大小
        if (h1, w1) != (h, w):
            left_image = cv2.resize(left_image, (w, h))
        if (h2, w2) != (h, w):
            right_image = cv2.resize(right_image, (w, h))
        
        # 水平拼接
        comparison = np.hstack((left_image, right_image))
        
        # 显示图片
        height, width = comparison.shape[:2]
        bytes_per_line = 3 * width
        comparison = cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB)
        q_image = QImage(comparison.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 等比例缩放以适应显示区域
        scaled_pixmap = pixmap.scaled(self.original_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_image_label.setPixmap(scaled_pixmap)
        
        # 更新图像信息
        self.update_comparison_info(left_image, right_image)

    def update_comparison_info(self, left_image, right_image):
        """更新对比图像的统计信息"""
        # 计算并显示直方图
        plt.figure(figsize=(5, 4))
        plt.subplot(121)
        self.plot_histogram(left_image, "原图")
        plt.subplot(122)
        self.plot_histogram(right_image, "处理后")
        
        # 将matplotlib图像转换为QPixmap
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        image = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
        plt.close()
        
        # 显示直方图
        pixmap = QPixmap.fromImage(image)
        self.hist_label.setPixmap(pixmap.scaled(
            self.hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 更新统计信息
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        info_text = "图像统计信息对比:\n"
        info_text += f"原图 -> 处理后\n"
        info_text += f"均值: {np.mean(left_gray):.2f} -> {np.mean(right_gray):.2f}\n"
        info_text += f"标准差: {np.std(left_gray):.2f} -> {np.std(right_gray):.2f}\n"
        info_text += f"中值: {np.median(left_gray):.2f} -> {np.median(right_gray):.2f}\n"
        
        # 计算梯度信息
        left_grad = self.calculate_gradient(left_gray)
        right_grad = self.calculate_gradient(right_gray)
        info_text += f"平均梯度: {left_grad:.2f} -> {right_grad:.2f}"
        
        self.hist_info.setText(info_text)

    def plot_histogram(self, image, title):
        """绘制单幅图像的直方图"""
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, alpha=0.7)
        plt.title(title)
        plt.xlim([0, 256])

    def calculate_gradient(self, gray_image):
        """计算图像的平均梯度"""
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        return np.mean(gradient_mag)

    def apply_histogram_equalization(self):
        """应用直方图均衡化"""
        if self.processor.current_image is None:
            return
        
        # 在LAB空间进行均衡化
        lab = cv2.cvtColor(self.processor.current_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge([l_eq, a, b])
        result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        
        # 更新显示
        self.update_result_display(result, 'histogram_eq')
        self.processor.current_image = result
        self.update_histogram()
        self.statusBar().showMessage('直方图均衡化完成')

    def update_image_info(self):
        """更新图像信息显示"""
        if self.processor.current_image is None:
            return
        
        # 计算并显示直方图
        self.update_histogram()
        
        # 计算并显示图像统计信息
        img = self.processor.current_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 基本统计信息
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        median_val = np.median(gray)
        
        # 梯度信息
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_mag)
        
        # 更新信息显示
        info_text = f"图像统计信息:\n"
        info_text += f"尺寸: {img.shape[1]}×{img.shape[0]}\n"
        info_text += f"均值: {mean_val:.2f}\n"
        info_text += f"标准差: {std_val:.2f}\n"
        info_text += f"中值: {median_val:.2f}\n"
        info_text += f"平均梯度: {mean_gradient:.2f}"
        self.hist_info.setText(info_text)

    def save_result(self):
        if self.processor.current_image is None:
            QMessageBox.warning(self, "警告", "没有可保存的图像！")
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "图片文件 (*.png *.jpg *.jpeg)"
        )
        if file_name:
            cv2.imwrite(file_name, self.processor.current_image)
            self.statusBar().showMessage('结果已保存')

    def update_brightness(self):
        """更新亮度"""
        if self.processor.current_image is None:
            return
        brightness = self.brightness_slider.value()
        self.brightness_value.setText(str(brightness))
        self.processor.brightness = brightness
        result = self.processor.adjust_brightness_contrast(brightness, self.processor.contrast)
        self.update_result_display(result, 'enhance')
        self.statusBar().showMessage(f'亮度: {brightness}')

    def update_contrast(self):
        """更新对比度"""
        if self.processor.current_image is None:
            return
        contrast = self.contrast_slider.value() / 100.0
        self.contrast_value.setText(str(contrast))
        self.processor.contrast = contrast
        result = self.processor.adjust_brightness_contrast(self.processor.brightness, contrast)
        self.update_result_display(result, 'enhance')
        self.statusBar().showMessage(f'对比度: {contrast:.2f}')

    def enhance_image(self):
        """图像增强"""
        if self.processor.current_image is None:
            return
        
        def operation():
            return self.processor.enhance_image()
        
        self.apply_operation(operation, "图像增强")

    def display_image(self, image):
        """显示单张图片（仅用于特殊情况的图像显示，不包含加载逻辑）"""
        if image is None:
            return
            
        # 转换图像格式
        height, width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(image_rgb.data, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 更新输入图像显示
        if 'input' in self.result_widgets:
            input_label = self.result_widgets['input']['label']
            scaled_pixmap = pixmap.scaled(
                input_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            input_label.setPixmap(scaled_pixmap)

    def load_directory(self):
        """批量处理文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if dir_path:
            # 创建进度对话框
            progress = QProgressDialog("处理图片中...", "取消", 0, 100, self)
            progress.setWindowModality(Qt.WindowModal)
            
            # 获取所有图片文件
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(dir_path, ext)))
            
            if not image_files:
                QMessageBox.warning(self, "警告", "所选文件夹中没有支持的图片文件！")
                return
            
            # 创建输出文件夹
            output_dir = os.path.join(dir_path, 'processed')
            os.makedirs(output_dir, exist_ok=True)
            
            # 处理每张图片
            for i, image_path in enumerate(image_files):
                try:
                    # 更新进度
                    progress.setValue(int((i / len(image_files)) * 100))
                    if progress.wasCanceled():
                        break
                    
                    # 处理图片
                    self.processor.load_image(image_path)
                    result, defects = self.processor.detect_defects_intelligent()
                    
                    # 保存结果
                    output_path = os.path.join(output_dir, 
                        f'processed_{os.path.basename(image_path)}')
                    # cv2.imwrite(output_path, result)
                    # 解决中文路径问题：使用 cv2.imencode 和内置 open 保存
                    _, buffer = cv2.imencode(os.path.splitext(output_path)[1], result)  # 编码为字节流
                    with open(output_path, 'wb') as f:
                        f.write(buffer)
                    
                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {str(e)}")
                    continue
            
            progress.setValue(100)
            QMessageBox.information(self, "完成", 
                f"批量处理完成！\n处理结果保存在: {output_dir}")

    def detect_defects(self):
        """检测所有缺陷"""
        if self.processor.original_image is None:
            return
        
        # 获取要处理的图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行检测
        result, defects = self.processor.detect_defects_intelligent()
        
        # 更新显示
        self.update_result_display(result, 'defect')
        
        # 显示检测结果
        result_text = f"检测结果:\n"
        result_text += f"裂缝: {len(defects['cracks'])} 处\n"
        result_text += f"坑洼: {len(defects['potholes'])} 处\n"
        result_text += f"积水: {len(defects['water'])} 处"
        self.result_text.setText(result_text)
        
        # 更新直方图和状态
        self.update_histogram()
        self.statusBar().showMessage('缺陷检测完成')

    def detect_edges(self):
        """边缘检测"""
        if self.processor.original_image is None:
            return
        
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行边缘检测
        result = self.processor.detect_edges()
        
        # 如果启用了边缘连接，则进行连接处理
        if self.edge_connect_checkbox.isChecked():
            min_threshold = self.min_threshold_slider.value()
            max_threshold = self.max_threshold_slider.value()
            # 将BGR转为灰度图
            gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
            # 进行边缘连接
            connected = self.processor.connect_edges(gray, min_threshold, max_threshold)
            # 转回BGR
            result = cv2.cvtColor(connected, cv2.COLOR_GRAY2BGR)
        
        # 更新显示
        self.update_result_display(result, 'edge')
        self.statusBar().showMessage('边缘检测完成')

    def apply_fft(self):
        """应用FFT滤波"""
        if self.processor.original_image is None:
            return
        
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行FFT滤波
        result = self.processor.fft_filter()
        
        # 更新显示
        self.update_result_display(result, 'fft')
        self.statusBar().showMessage('FFT滤波完成')

    def detect_cracks_only(self):
        """仅检测裂缝"""
        if self.processor.original_image is None:
            return
        
        # 获取要处理的图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行检测
        result, defects = self.processor.detect_defects_intelligent()
        
        # 更新显示
        self.update_result_display(result, 'defect')
        self.result_text.setText(f"检测到 {len(defects['cracks'])} 处裂缝")
        
        # 更新直方图和状态
        self.update_histogram()
        self.statusBar().showMessage('裂缝检测完成')

    def detect_potholes_only(self):
        """仅检测坑洼"""
        if self.processor.original_image is None:
            return
        
        # 获取要处理的图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行检测
        result, defects = self.processor.detect_defects_intelligent()
        
        # 更新显示
        self.update_result_display(result, 'defect')
        self.result_text.setText(f"检测到 {len(defects['potholes'])} 处坑洼")
        
        # 更新直方图和状态
        self.update_histogram()
        self.statusBar().showMessage('坑洼检测完成')

    def detect_water_only(self):
        """仅检测积水"""
        if self.processor.original_image is None:
            return
        
        # 获取要处理的图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行检测
        result, defects = self.processor.detect_defects_intelligent()
        
        # 更新显示
        self.update_result_display(result, 'defect')
        self.result_text.setText(f"检测到 {len(defects['water'])} 处积水")
        
        # 更新直方图和状态
        self.update_histogram()
        self.statusBar().showMessage('积水检测完成')

    def update_histogram(self):
        """更新直方图显示"""
        if self.processor.current_image is None:
            return
        
        # 获取要显示直方图的图像和来源信息
        display_image = None
        source_info = "当前显示: "
        
        # 检查是否有选中的图像
        for key, result in self.result_widgets.items():
            if result['selected'] and result['has_result']:
                display_image = self.pixmap_to_cv2(result['label'].pixmap())
                source_info += f"{result['widget'].findChild(QLabel).text()}"
                break
        
        # 如果没有选中的图像，使用最后一次处理的图像
        if display_image is None:
            display_image = self.processor.current_image
            # 查找最后一次处理的窗口
            for key, result in reversed(list(self.result_widgets.items())):
                if result['has_result'] and key != 'input':
                    source_info += f"{result['widget'].findChild(QLabel).text()}"
                    break
            else:
                source_info += "原始图像"
        
        # 更新直方图和图像信息
        self.update_clicked_image_info(display_image, source_info.split(": ")[1])

    def apply_morph_op(self, op_type):
        """应用形态学操作"""
        if self.processor.original_image is None:
            return
        
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        kernel = np.ones((self.processor.morph_size, self.processor.morph_size), np.uint8)
        
        # 执行相应的形态学操作
        if op_type == 'erode':
            result = cv2.erode(source_image, kernel, iterations=1)
            op_name = '腐蚀'
            display_type = 'morph_erode'
        elif op_type == 'dilate':
            result = cv2.dilate(source_image, kernel, iterations=1)
            op_name = '膨胀'
            display_type = 'morph_dilate'
        elif op_type == 'open':
            result = cv2.morphologyEx(source_image, cv2.MORPH_OPEN, kernel)
            op_name = '开运算'
            display_type = 'morph_open'
        elif op_type == 'close':
            result = cv2.morphologyEx(source_image, cv2.MORPH_CLOSE, kernel)
            op_name = '闭运算'
            display_type = 'morph_close'
        elif op_type == 'gradient':
            result = cv2.morphologyEx(source_image, cv2.MORPH_GRADIENT, kernel)
            op_name = '形态学梯度'
            display_type = 'morph_gradient'
        
        # 更新显示
        self.update_result_display(result, display_type)
        
        # 更新直方图
        self.update_histogram()
        self.statusBar().showMessage(f'{op_name}完成')

    def add_to_history(self, image, operation_name):
        """添加处理结果到历史记录"""
        # 创建历史记录项
        item = QWidget()
        item_layout = QVBoxLayout(item)
        
        # 添加操作名称
        name_label = QLabel(operation_name)
        name_label.setStyleSheet("font-weight: bold;")
        item_layout.addWidget(name_label)
        
        # 添加缩略图
        thumb_label = QLabel()
        thumb_label.setFixedSize(280, 200)
        thumb_label.setAlignment(Qt.AlignCenter)
        
        # 创建缩略图
        height, width = image.shape[:2]
        thumb = cv2.resize(image, (280, int(280 * height / width)))
        thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
        h, w = thumb.shape[:2]
        q_image = QImage(thumb.data, w, h, w * 3, QImage.Format_RGB888)
        thumb_label.setPixmap(QPixmap.fromImage(q_image))
        
        item_layout.addWidget(thumb_label)
        
        # 添加查看按钮
        view_btn = QPushButton("查看结果")
        view_btn.clicked.connect(lambda: self.display_comparison(
            self.processor.original_image, image))
        item_layout.addWidget(view_btn)
        
        # 添加分割线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        
        # 将项目添加到历史记录开头
        self.history_list.insertWidget(0, item)
        self.history_list.insertWidget(1, line)

    def apply_operation(self, operation_func, operation_name):
        """通用的操作应用函数"""
        if self.processor.original_image is None:
            return
        
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        # 执行操作
        result = operation_func()
        
        # 更新显示
        operation_type_map = {
            '边缘检测': 'edge',
            'FFT滤波': 'fft',
            '图像增强': 'enhance',
            '直方图均衡化': 'histogram_eq',
            '自适应直方图均衡化': 'clahe'
        }
        
        operation_type = operation_type_map.get(operation_name, 'enhance')
        self.update_result_display(result, operation_type)
        
        # 更新当前图像
        self.processor.current_image = result
        
        # 更新直方图
        self.update_histogram()
        
        # 更新状态栏
        self.statusBar().showMessage(f'{operation_name}完成')

    def apply_clahe(self):
        """应用自适应直方图均衡化"""
        if self.processor.current_image is None:
            return
        
        def operation():
            # 转换到LAB空间
            lab = cv2.cvtColor(self.processor.current_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # 创建CLAHE对象
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            
            # 应用CLAHE到L通道
            l_clahe = clahe.apply(l)
            
            # 合并通道
            lab_clahe = cv2.merge([l_clahe, a, b])
            
            # 转换回BGR
            return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        
        self.apply_operation(operation, "自适应直方图均衡化")

    def show_detailed_info(self):
        """显示所有处理结果的详细信息"""
        if self.processor.original_image is None:
            return
        
        # 创建详细信息窗口
        dialog = QDialog(self)
        dialog.setWindowTitle("图像处理详细信息")
        dialog.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        
        # 遍历所有有结果的图像
        for key, result in self.result_widgets.items():
            if result['has_result'] or key == 'input':
                # 创建组
                group = QGroupBox(result['widget'].findChild(QLabel).text())
                group_layout = QHBoxLayout()
                
                # 获取图像数据
                pixmap = result['label'].pixmap()
                if pixmap:
                    image = self.pixmap_to_cv2(pixmap)
                    
                    # 创建直方图
                    fig = plt.figure(figsize=(4, 3))
                    colors = ('b', 'g', 'r')
                    for i, color in enumerate(colors):
                        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                        plt.plot(hist, color=color)
                    plt.title('RGB直方图')
                    plt.xlabel('像素值')
                    plt.ylabel('频率')
                    plt.legend()
                    plt.grid(True)
                    
                    # 转换直方图为QLabel
                    canvas = FigureCanvas(fig)
                    canvas.draw()
                    hist_label = QLabel()
                    hist_label.setPixmap(QPixmap.fromImage(QImage(
                        canvas.buffer_rgba(), 
                        int(canvas.width()), 
                        int(canvas.height()), 
                        QImage.Format_RGBA8888
                    )))
                    plt.close()
                    
                    # 添加统计信息
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    info_text = f"统计信息:\n"
                    info_text += f"均值: {np.mean(gray):.2f}\n"
                    info_text += f"标准差: {np.std(gray):.2f}\n"
                    info_text += f"中值: {np.median(gray):.2f}\n"
                    info_text += f"最小值: {np.min(gray):.2f}\n"
                    info_text += f"最大值: {np.max(gray):.2f}"
                    
                    info_label = QLabel(info_text)
                    info_label.setStyleSheet("font-size: 12px;")
                    
                    group_layout.addWidget(hist_label)
                    group_layout.addWidget(info_label)
                
                group.setLayout(group_layout)
                content_layout.addWidget(group)
        
        scroll.setWidget(content)
        layout.addWidget(scroll)
        
        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def pixmap_to_qimage(self, pixmap):
        """将QPixmap转换为QImage"""
        qimage = pixmap.toImage()
        return qimage

    def update_image_info(self):
        """更新图像信息显示"""
        if self.processor.current_image is None:
            return
        
        # 计算并显示直方图
        self.update_histogram()
        
        # 计算并显示图像统计信息
        img = self.processor.current_image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 基本统计信息
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        median_val = np.median(gray)
        
        # 梯度信息
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        mean_gradient = np.mean(gradient_mag)
        
        # 更新信息显示
        info_text = f"图像统计信息:\n"
        info_text += f"尺寸: {img.shape[1]}×{img.shape[0]}\n"
        info_text += f"均值: {mean_val:.2f}\n"
        info_text += f"标准差: {std_val:.2f}\n"
        info_text += f"中值: {median_val:.2f}\n"
        info_text += f"平均梯度: {mean_gradient:.2f}"
        return info_text

    def update_canny_low(self):
        """更新Canny边缘检测的低阈值"""
        if self.processor.current_image is None:
            return
        value = self.canny_low_slider.value()
        self.canny_low_value.setText(str(value))
        self.processor.canny_low = value
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        # 执行边缘检测
        result = self.processor.detect_edges()
        # 更新显示
        self.update_result_display(result, 'edge')
        self.statusBar().showMessage(f'Canny低阈值: {value}')

    def update_canny_high(self):
        """更新Canny边缘检测的高阈值"""
        if self.processor.current_image is None:
            return
        value = self.canny_high_slider.value()
        self.canny_high_value.setText(str(value))
        self.processor.canny_high = value
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        # 执行边缘检测
        result = self.processor.detect_edges()
        # 更新显示
        self.update_result_display(result, 'edge')
        self.statusBar().showMessage(f'Canny高阈值: {value}')

    def update_fft_radius(self):
        """更新FFT滤波半径"""
        if self.processor.current_image is None:
            return
        value = self.fft_radius_slider.value()
        self.fft_radius_value.setText(str(value))
        self.processor.fft_radius = value
        # 获取当前处理的源图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        # 执行FFT滤波
        result = self.processor.fft_filter()
        # 更新显示
        self.update_result_display(result, 'fft')
        self.statusBar().showMessage(f'FFT半径: {value}')

    def update_morph_size(self):
        """更新形态学操作的核大小"""
        if self.processor.current_image is None:
            return
        value = self.morph_size_slider.value()
        self.morph_size_value.setText(str(value))
        self.processor.morph_size = value
        self.statusBar().showMessage(f'形态学核大小: {value}')

    def show_image_viewer(self, label):
        """显示图像查看器对话框"""
        if not label.pixmap():
            return
        
        # 将QPixmap转换为OpenCV图像
        image = self.pixmap_to_cv2(label.pixmap())
        if image is not None:
            dialog = ImageViewerDialog(image, self)
            dialog.exec_()

    def detect_defects_ai(self):
        """使用AI方法进行缺陷检测"""
        if self.processor.original_image is None:
            return
        
        # 获取要处理的图像
        source_image = self.get_current_source_image()
        self.processor.current_image = source_image
        
        try:
            # 创建进度对话框
            progress = QProgressDialog("正在进行AI检测...", "取消", 0, 100, self)
            progress.setWindowTitle("AI检测进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(10)
            QApplication.processEvents()
            
            # 设置检测模式
            if self.bbox_radio.isChecked():
                self.processor.detection_mode = 'bbox'
            elif self.segment_radio.isChecked():
                self.processor.detection_mode = 'segment'
            else:
                self.processor.detection_mode = 'both'
            
            # 尝试加载模型
            if self.processor.detection_mode in ['bbox', 'both'] and self.processor.yolo_model is None:
                progress.setLabelText("正在加载边界框检测模型...")
                progress.setValue(20)
                QApplication.processEvents()
                
                if not self.processor.load_yolo_model():
                    progress.close()
                    QMessageBox.warning(self, "警告", "边界框检测模型加载失败，请检查模型文件是否存在")
                    return
            
            if self.processor.detection_mode in ['segment', 'both'] and self.processor.segment_model is None:
                progress.setLabelText("正在加载分割模型...")
                progress.setValue(30)
                QApplication.processEvents()
                
                if not self.processor.load_segment_model():
                    progress.close()
                    QMessageBox.warning(self, "警告", "分割模型加载失败，请检查模型文件是否存在")
                    return
            
            progress.setLabelText("正在进行目标检测...")
            progress.setValue(50)
            QApplication.processEvents()
            
            # 执行AI检测
            result, defects = self.processor.detect_defects_ai()
            
            if progress.wasCanceled():
                progress.close()
                return
            
            progress.setLabelText("正在更新显示...")
            progress.setValue(80)
            QApplication.processEvents()
            
            # 更新显示
            self.update_result_display(result, 'defect')
            
            # 显示检测结果
            result_text = f"AI检测结果:\n"
            
            # 显示边界框检测结果
            if self.processor.detection_mode in ['bbox', 'both']:
                bbox_stats = defects['stats']['bbox']
                result_text += f"\n边界框检测:\n"
                result_text += f"- 检测到坑洼: {bbox_stats['count']} 处\n"
                if bbox_stats['count'] > 0:
                    result_text += "- 各区域面积(像素):\n"
                    for i, area in enumerate(bbox_stats['areas'], 1):
                        result_text += f"  区域{i}: {area}\n"
                    result_text += f"- 总检测区域: {sum(bbox_stats['areas'])} 像素\n"
            
            # 显示分割检测结果
            if self.processor.detection_mode in ['segment', 'both']:
                segment_stats = defects['stats']['segment']
                result_text += f"\n分割检测:\n"
                result_text += f"- 检测到目标: {segment_stats['count']} 处\n"
                if segment_stats['count'] > 0:
                    result_text += "- 各区域掩码面积(像素):\n"
                    for i, area in enumerate(segment_stats['areas'], 1):
                        result_text += f"  区域{i}: {area}\n"
                    result_text += f"- 总掩码面积: {sum(segment_stats['areas'])} 像素\n"
            
            self.result_text.setText(result_text)
            
            # 更新直方图和状态
            self.update_histogram()
            
            progress.close()
            self.statusBar().showMessage('AI检测完成')
            
        except ImportError:
            QMessageBox.warning(self, "警告", "未安装ultralytics库，无法使用AI检测功能")

    def batch_process(self):
        """批量处理图片"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not dir_path:  # 如果用户取消选择，直接返回
            return
            
        # 获取所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(dir_path, ext)))
        
        if not image_files:
            QMessageBox.warning(self, "警告", "所选文件夹中没有支持的图片文件！")
            return
            
        # 保存当前图像处理器的状态
        saved_original_image = self.processor.original_image
        saved_current_image = self.processor.current_image
            
        # 创建进度对话框
        progress = QProgressDialog("准备处理...", "取消", 0, len(image_files), self)
        progress.setWindowTitle("批处理进度")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        
        # 获取选择的处理方法和模式
        method = self.process_method.currentText()
        
        # 如果是AI方法，设置检测模式
        if method == "AI方法":
            mode = self.ai_mode.currentText()
            if mode == "边界框检测":
                self.processor.detection_mode = 'bbox'
            elif mode == "分割检测":
                self.processor.detection_mode = 'segment'
            else:
                self.processor.detection_mode = 'both'
        
        # 创建输出文件夹
        output_dir = os.path.join(dir_path, f'processed_{method.lower().replace("方法", "")}')
        if method == "AI方法":
            output_dir = os.path.join(output_dir, f'{self.processor.detection_mode.lower()}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 如果是AI方法，预先加载模型
        if method == "AI方法":
            if self.processor.detection_mode in ['bbox', 'both'] and self.processor.yolo_model is None:
                progress.setLabelText("正在加载边界框检测模型...")
                progress.setValue(0)
                QApplication.processEvents()
                
                if not self.processor.load_yolo_model():
                    QMessageBox.warning(self, "警告", "边界框检测模型加载失败，请检查模型文件是否存在")
                    # 恢复状态
                    self.processor.original_image = saved_original_image
                    self.processor.current_image = saved_current_image
                    return
                    
            if self.processor.detection_mode in ['segment', 'both'] and self.processor.segment_model is None:
                progress.setLabelText("正在加载分割模型...")
                progress.setValue(0)
                QApplication.processEvents()
                
                if not self.processor.load_segment_model():
                    QMessageBox.warning(self, "警告", "分割模型加载失败，请检查模型文件是否存在")
                    # 恢复状态
                    self.processor.original_image = saved_original_image
                    self.processor.current_image = saved_current_image
                    return
        
        # 处理每张图片
        processed_count = 0
        try:
            for i, image_path in enumerate(image_files):
                if progress.wasCanceled():  # 如果用户取消，直接退出循环
                    break
                    
                try:
                    # 更新进度
                    progress.setValue(i)
                    progress.setLabelText(f"正在处理: {os.path.basename(image_path)}\n"
                                        f"已完成: {processed_count}/{len(image_files)}\n"
                                        f"当前进度: {int((i/len(image_files))*100)}%")
                    QApplication.processEvents()
                    
                    # 加载图片
                    self.processor.load_image(image_path)
                    
                    # 根据选择的方法进行处理
                    if method == "AI方法":
                        result, defects = self.processor.detect_defects_ai()
                    else:
                        result, defects = self.processor.detect_defects_intelligent()
                    
                    # 保存结果
                    output_path = os.path.join(output_dir, f'processed_{os.path.basename(image_path)}')
                    _, buffer = cv2.imencode(os.path.splitext(output_path)[1], result)
                    with open(output_path, 'wb') as f:
                        f.write(buffer)
                    
                    # 保存检测结果信息
                    info_path = os.path.splitext(output_path)[0] + '_info.txt'
                    with open(info_path, 'w', encoding='utf-8') as f:
                        f.write(f"检测方法: {method}\n")
                        if method == "AI方法":
                            f.write(f"检测模式: {mode}\n")
                        f.write(f"检测结果:\n")
                        if method == "AI方法":
                            if 'stats' in defects:
                                if self.processor.detection_mode in ['bbox', 'both']:
                                    bbox_stats = defects['stats']['bbox']
                                    f.write(f"边界框检测:\n")
                                    f.write(f"- 检测到坑洼: {bbox_stats['count']} 处\n")
                                    if bbox_stats['count'] > 0:
                                        f.write("- 各区域面积(像素):\n")
                                        for i, area in enumerate(bbox_stats['areas'], 1):
                                            f.write(f"  区域{i}: {area}\n")
                                
                                if self.processor.detection_mode in ['segment', 'both']:
                                    segment_stats = defects['stats']['segment']
                                    f.write(f"\n分割检测:\n")
                                    f.write(f"- 检测到目标: {segment_stats['count']} 处\n")
                                    if segment_stats['count'] > 0:
                                        f.write("- 各区域掩码面积(像素):\n")
                                        for i, area in enumerate(segment_stats['areas'], 1):
                                            f.write(f"  区域{i}: {area}\n")
                            else:
                                f.write(f"坑洼: {len(defects['potholes'])} 处\n")
                        else:
                            f.write(f"裂缝: {len(defects['cracks'])} 处\n")
                            f.write(f"坑洼: {len(defects['potholes'])} 处\n")
                            f.write(f"积水: {len(defects['water'])} 处\n")
                    
                    processed_count += 1
                    
                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {str(e)}")
                    continue
            
        finally:
            # 恢复原始状态
            self.processor.original_image = saved_original_image
            self.processor.current_image = saved_current_image
            
            progress.close()  # 关闭进度对话框
            
            if processed_count > 0:  # 只有在实际处理了图片时才显示完成消息
                QMessageBox.information(self, "完成", 
                    f"批量处理完成！\n"
                    f"成功处理: {processed_count}/{len(image_files)} 张图片\n"
                    f"处理结果保存在: {output_dir}")

    def copy_selected_image(self):
        """复制选中的图像到剪贴板"""
        # 查找选中的图像
        selected_image = None
        for result in self.result_widgets.values():
            if result['selected'] and result['has_result']:
                pixmap = result['label'].pixmap()
                if pixmap:
                    selected_image = pixmap
                    break
        
        if selected_image:
            clipboard = QApplication.clipboard()
            clipboard.setPixmap(selected_image)
            self.statusBar().showMessage('图像已复制到剪贴板')
        else:
            self.statusBar().showMessage('没有选中的图像可复制')

    def on_process_method_changed(self, text):
        """处理方法改变时的响应"""
        self.ai_mode.setEnabled(text == "AI方法")

    def on_edge_connect_changed(self, state):
        """处理边缘连接启用状态改变"""
        enabled = state == Qt.Checked
        self.min_threshold_slider.setEnabled(enabled)
        self.max_threshold_slider.setEnabled(enabled)
        if self.processor.current_image is not None:
            self.detect_edges()  # 重新执行边缘检测
    
    def update_min_threshold(self):
        """更新最小连接阈值"""
        if self.processor.current_image is None:
            return
        value = self.min_threshold_slider.value()
        self.min_threshold_value.setText(str(value))
        if value >= self.max_threshold_slider.value():
            self.max_threshold_slider.setValue(value + 1)
        self.detect_edges()  # 重新执行边缘检测
    
    def update_max_threshold(self):
        """更新最大连接阈值"""
        if self.processor.current_image is None:
            return
        value = self.max_threshold_slider.value()
        self.max_threshold_value.setText(str(value))
        if value <= self.min_threshold_slider.value():
            self.min_threshold_slider.setValue(value - 1)
        self.detect_edges()  # 重新执行边缘检测

class ImageViewerDialog(QDialog):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self.selected_rect = None
        self.moving_edge = None
        self.last_pos = None
        self.rectangles = []  # [(rect, id), ...]
        self.next_rect_id = 1
        self.current_scale = 1.0  # 添加缩放比例跟踪
        
        # 设置窗口属性
        self.setWindowTitle('图像查看器')
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        self.initUI()
        
        # 初始化图像显示
        self.setup_image()

    def setup_image(self):
        """初始化图像显示"""
        height, width = self.image.shape[:2]
        bytes_per_line = 3 * width
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        q_image = QImage(image_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.image_item = self.image_scene.addPixmap(pixmap)
        self.image_scene.setSceneRect(self.image_item.boundingRect())

    def eventFilter(self, source, event):
        """事件过滤器"""
        if source == self.image_view.viewport():
            if event.type() == QEvent.MouseButtonPress:
                return self.handle_mouse_press(event)
            elif event.type() == QEvent.MouseMove:
                return self.handle_mouse_move(event)
            elif event.type() == QEvent.MouseButtonRelease:
                return self.handle_mouse_release(event)
            elif event.type() == QEvent.Wheel and event.modifiers() == Qt.ControlModifier:
                return self.handle_wheel_event(event)
        return super().eventFilter(source, event)

    def handle_mouse_press(self, event):
        """处理鼠标按下事件"""
        pos = self.image_view.mapToScene(event.pos())
        
        # 检查是否点击了现有矩形的边缘
        for rect_tuple in self.rectangles:
            edge = self.get_clicked_edge(pos, rect_tuple)
            if edge:
                self.moving_edge = edge
                self.selected_rect = rect_tuple
                self.last_pos = pos
                return True
        
        # 如果没有点击边缘，开始绘制新矩形
        self.rect_start = pos
        self.drawing = True
        return True

    def handle_mouse_move(self, event):
        """处理鼠标移动事件"""
        pos = self.image_view.mapToScene(event.pos())
        
        # 更新位置显示
        if self.image_scene.sceneRect().contains(pos):
            self.position_label.setText(f"位置: ({int(pos.x())}, {int(pos.y())})")
        
        if self.drawing:
            # 绘制新矩形
            self.rect_end = pos
            self.update_temp_rectangle()
        elif self.moving_edge and self.selected_rect:
            # 移动矩形边缘
            new_rect = self.move_rectangle_edge(pos)
            if new_rect:
                rect_id = self.selected_rect[1]
                self.selected_rect = (new_rect, rect_id)
                self.update_scene()
                self.update_roi_result(new_rect, rect_id)
        return True

    def handle_mouse_release(self, event):
        """处理鼠标释放事件"""
        if self.drawing:
            self.drawing = False
            if self.rect_start and self.rect_end:
                rect = QRectF(self.rect_start, self.rect_end).normalized()
                if rect.width() > 5 and rect.height() > 5:
                    # 添加矩形和ID
                    self.rectangles.append((rect, self.next_rect_id))
                    self.add_roi_result(rect)
                    self.next_rect_id += 1
                    self.update_scene()
            self.rect_start = None
            self.rect_end = None
        
        self.moving_edge = None
        self.selected_rect = None
        self.last_pos = None
        return True

    def handle_wheel_event(self, event):
        """处理鼠标滚轮事件"""
        if event.angleDelta().y() > 0:
            self.zoom(1.2)
        else:
            self.zoom(0.8)
        return True

    def get_clicked_edge(self, pos, rect_tuple, threshold=5.0):
        """检测是否点击了矩形的边缘"""
        if not rect_tuple:
            return None
        
        rect, _ = rect_tuple  # 从元组中解包出 QRectF 对象
        
        # 检查每条边
        left = abs(pos.x() - rect.left())
        right = abs(pos.x() - rect.right())
        top = abs(pos.y() - rect.top())
        bottom = abs(pos.y() - rect.bottom())
        
        # 判断点击位置是否在矩形边界上
        if left < threshold and rect.top() <= pos.y() <= rect.bottom():
            return 'left'
        if right < threshold and rect.top() <= pos.y() <= rect.bottom():
            return 'right'
        if top < threshold and rect.left() <= pos.x() <= rect.right():
            return 'top'
        if bottom < threshold and rect.left() <= pos.x() <= rect.right():
            return 'bottom'
        
        return None

    def zoom(self, factor):
        """缩放视图"""
        self.current_scale *= factor
        self.image_view.scale(factor, factor)
        # 更新状态栏显示当前缩放比例
        self.zoom_label.setText(f"缩放: {self.current_scale:.1f}x")

    def zoom_fit(self):
        """适应窗口大小"""
        self.image_view.fitInView(self.image_scene.sceneRect(), Qt.KeepAspectRatio)
        # 重置缩放比例
        self.current_scale = 1.0
        self.zoom_label.setText("缩放: 1.0x")

    def zoom_actual(self):
        """实际大小"""
        self.image_view.resetTransform()
        self.current_scale = 1.0
        self.zoom_label.setText("缩放: 1.0x")

    def initUI(self):
        # 创建主布局
        layout = QHBoxLayout(self)
        layout.setSpacing(10)
        
        # 左侧图像显示区域
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(5)
        
        # 添加缩放控制工具栏
        toolbar = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_out_btn = QPushButton("缩小")
        self.zoom_fit_btn = QPushButton("适应窗口")
        self.zoom_actual_btn = QPushButton("实际大小")
        self.zoom_label = QLabel("缩放: 1.0x")
        
        self.zoom_in_btn.clicked.connect(lambda: self.zoom(1.2))
        self.zoom_out_btn.clicked.connect(lambda: self.zoom(0.8))
        self.zoom_fit_btn.clicked.connect(self.zoom_fit)
        self.zoom_actual_btn.clicked.connect(self.zoom_actual)
        
        toolbar.addWidget(self.zoom_in_btn)
        toolbar.addWidget(self.zoom_out_btn)
        toolbar.addWidget(self.zoom_fit_btn)
        toolbar.addWidget(self.zoom_actual_btn)
        toolbar.addWidget(self.zoom_label)
        toolbar.addStretch()
        
        left_layout.addLayout(toolbar)
        
        # 图像显示区域
        self.image_scene = QGraphicsScene()
        self.image_view = QGraphicsView(self.image_scene)
        self.image_view.setRenderHint(QPainter.Antialiasing)
        self.image_view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.image_view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        
        # 添加鼠标追踪
        self.image_view.viewport().setMouseTracking(True)
        self.image_view.setDragMode(QGraphicsView.ScrollHandDrag)  # 允许拖动
        
        left_layout.addWidget(self.image_view)
        
        # 添加状态栏
        status_bar = QHBoxLayout()
        self.position_label = QLabel("位置: -")
        status_bar.addWidget(self.position_label)
        status_bar.addStretch()
        left_layout.addLayout(status_bar)
        
        # 右侧信息面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(10)
        
        # 添加控制按钮
        button_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清除所有")
        self.undo_btn = QPushButton("撤销上一个")
        self.clear_btn.clicked.connect(self.clear_rectangles)
        self.undo_btn.clicked.connect(self.undo_last_rectangle)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.undo_btn)
        right_layout.addLayout(button_layout)
        
        # 添加滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # 创建结果容器
        self.results_container = QWidget()
        self.results_layout = QVBoxLayout(self.results_container)
        self.results_layout.setSpacing(15)
        scroll.setWidget(self.results_container)
        
        right_layout.addWidget(scroll)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        
        # 设置分割器的初始大小比例
        splitter.setSizes([int(self.width() * 0.6), int(self.width() * 0.4)])
        
        # 设置右侧面板的最小宽度
        right_panel.setMinimumWidth(500)
        
        layout.addWidget(splitter)
        
        # 安装事件过滤器
        self.image_view.viewport().installEventFilter(self)
        
        # 添加缩放支持
        self.image_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.image_view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.image_view.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.image_view.setFrameShape(QFrame.NoFrame)

    def add_roi_result(self, rect):
        """添加ROI区域分析结果"""
        # 创建结果组件
        result_widget = QGroupBox(f"区域 #{self.next_rect_id}")
        result_widget.setMinimumHeight(300)
        layout = QHBoxLayout(result_widget)
        layout.setSpacing(15)
        
        # 左侧创建直方图容器
        hist_container = QWidget()
        hist_layout = QVBoxLayout(hist_container)
        hist_layout.setSpacing(5)
        
        # 添加直方图
        hist_label = QLabel()
        hist_label.setMinimumSize(400, 300)
        
        # 添加放大按钮
        zoom_btn = QPushButton("放大查看")
        zoom_btn.clicked.connect(lambda: self.show_zoomed_histogram(hist_label))
        
        hist_layout.addWidget(hist_label)
        hist_layout.addWidget(zoom_btn)
        
        # 右侧添加统计信息
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMinimumWidth(400)
        
        stats_container = QWidget()
        stats_label = QLabel()
        stats_label.setWordWrap(True)
        stats_label.setAlignment(Qt.AlignTop)
        
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.addWidget(stats_label)
        stats_scroll.setWidget(stats_container)
        
        # 添加到主布局
        layout.addWidget(hist_container)
        layout.addWidget(stats_scroll)
        
        # 更新ROI信息
        self.update_roi_info(rect, hist_label, stats_label, self.next_rect_id - 1)
        
        # 添加到结果容器
        self.results_layout.insertWidget(0, result_widget)

    def show_zoomed_histogram(self, hist_label):
        """显示放大的直方图"""
        if not hist_label.pixmap():
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("直方图详细查看")
        dialog.setMinimumSize(800, 600)
        dialog.resize(1000, 800)  # 设置默认大小
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowMaximizeButtonHint | Qt.WindowMinimizeButtonHint)
        
        layout = QVBoxLayout(dialog)
        layout.setSpacing(10)
        
        # 创建滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # 创建新的标签显示放大的直方图
        zoomed_label = QLabel()
        zoomed_label.setPixmap(hist_label.pixmap().scaled(
            800, 600,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))
        
        scroll.setWidget(zoomed_label)
        layout.addWidget(scroll)
        
        # 添加缩放控制
        zoom_layout = QHBoxLayout()
        zoom_in = QPushButton("放大")
        zoom_out = QPushButton("缩小")
        zoom_fit = QPushButton("适应窗口")
        
        current_scale = 1.0
        
        def zoom(factor):
            nonlocal current_scale
            current_scale *= factor
            new_size = hist_label.pixmap().size() * current_scale
            zoomed_label.setPixmap(hist_label.pixmap().scaled(
                new_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
        
        zoom_in.clicked.connect(lambda: zoom(1.2))
        zoom_out.clicked.connect(lambda: zoom(0.8))
        zoom_fit.clicked.connect(lambda: zoomed_label.setPixmap(hist_label.pixmap().scaled(
            scroll.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )))
        
        zoom_layout.addWidget(zoom_in)
        zoom_layout.addWidget(zoom_out)
        zoom_layout.addWidget(zoom_fit)
        zoom_layout.addStretch()
        
        layout.addLayout(zoom_layout)
        
        # 添加关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.exec_()

    def move_rectangle_edge(self, pos):
        """移动矩形边缘，返回新的矩形"""
        if not self.selected_rect or not self.moving_edge:
            return None
        
        rect = self.selected_rect[0]  # 获取QRectF对象
        new_rect = QRectF(rect)
        
        if self.moving_edge == 'left':
            new_rect.setLeft(pos.x())
        elif self.moving_edge == 'right':
            new_rect.setRight(pos.x())
        elif self.moving_edge == 'top':
            new_rect.setTop(pos.y())
        elif self.moving_edge == 'bottom':
            new_rect.setBottom(pos.y())
        
        # 确保矩形大小合法
        if new_rect.width() > 5 and new_rect.height() > 5:
            return new_rect
        return None

    def update_scene(self):
        """更新场景中的所有矩形"""
        # 清除现有项目（保留图像）
        for item in self.image_scene.items():
            if isinstance(item, (QGraphicsRectItem, QGraphicsTextItem)):
                self.image_scene.removeItem(item)
        
        # 重新绘制所有矩形和编号
        for rect, rect_id in self.rectangles:
            self.add_rectangle(rect, rect_id)
    
    def add_rectangle(self, rect, rect_id):
        """添加带编号的矩形到场景"""
        # 添加矩形
        rect_item = QGraphicsRectItem(rect)
        rect_item.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        self.image_scene.addItem(rect_item)
        
        # 添加编号文本
        text_item = QGraphicsTextItem(str(rect_id))
        text_item.setDefaultTextColor(Qt.red)
        text_item.setPos(rect.topLeft())
        self.image_scene.addItem(text_item)

    def clear_rectangles(self):
        """清除所有矩形"""
        self.rectangles.clear()
        self.next_rect_id = 1
        self.update_scene()
        
        # 清除所有结果
        while self.results_layout.count():
            item = self.results_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def undo_last_rectangle(self):
        """撤销最后一个矩形"""
        if self.rectangles:
            self.rectangles.pop()
            self.update_scene()
            
            # 移除最后一个结果
            if self.results_layout.count() > 0:
                item = self.results_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

    def update_temp_rectangle(self):
        """更新临时矩形"""
        self.update_scene()
        if self.rect_start and self.rect_end:
            rect = QRectF(self.rect_start, self.rect_end).normalized()
            rect_item = QGraphicsRectItem(rect)
            rect_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
            self.image_scene.addItem(rect_item)
            
            # 添加临时编号文本
            text_item = QGraphicsTextItem(str(self.next_rect_id))
            text_item.setDefaultTextColor(Qt.red)
            text_item.setPos(rect.topLeft())
            self.image_scene.addItem(text_item)

    def update_roi_result(self, rect, rect_id):
        """更新指定ID的ROI结果"""
        # 查找对应的结果组件
        for i in range(self.results_layout.count()):
            widget = self.results_layout.itemAt(i).widget()
            if isinstance(widget, QGroupBox) and widget.title() == f"区域 #{rect_id}":
                # 更新直方图和统计信息
                hist_label = widget.findChild(QLabel)
                stats_label = widget.findChildren(QLabel)[1]  # 第二个QLabel是统计信息
                self.update_roi_info(rect, hist_label, stats_label, rect_id)
                break

    def update_roi_info(self, rect, hist_label, stats_label, rect_id=None):
        """更新ROI区域的信息"""
        if not rect:
            return
        
        # 如果没有提供rect_id，使用next_rect_id-1
        if rect_id is None:
            rect_id = self.next_rect_id - 1
        
        # 获取矩形区域在图像中的坐标
        x1, y1 = int(rect.left()), int(rect.top())
        x2, y2 = int(rect.right()), int(rect.bottom())
        
        # 确保坐标在图像范围内
        height, width = self.image.shape[:2]
        x1 = max(0, min(x1, width-1))
        x2 = max(0, min(x2, width-1))
        y1 = max(0, min(y1, height-1))
        y2 = max(0, min(y2, height-1))
        
        # 提取ROI区域
        roi = self.image[y1:y2+1, x1:x2+1]
        
        if roi.size == 0:
            return
        
        # 创建直方图
        plt.figure(figsize=(4, 3))
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([roi], [i], None, [256], [0, 256])
            plt.plot(hist, color=color, label=f'{color.upper()}通道')
        
        plt.title(f'区域 #{rect_id+1} RGB直方图')
        plt.xlabel('像素值')
        plt.ylabel('频率')
        plt.legend()
        plt.grid(True)
        
        # 将matplotlib图像转换为QPixmap
        canvas = FigureCanvas(plt.gcf())
        canvas.draw()
        width, height = canvas.get_width_height()
        image = QImage(canvas.buffer_rgba(), width, height, QImage.Format_RGBA8888)
        plt.close()
        
        # 显示直方图
        pixmap = QPixmap.fromImage(image)
        hist_label.setPixmap(pixmap.scaled(
            hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        
        # 更新统计信息
        stats_text = f"<b>区域 #{rect_id+1} 统计信息:</b><br>"
        stats_text += f"位置: ({x1}, {y1}) - ({x2}, {y2})<br>"
        stats_text += f"大小: {x2-x1+1} × {y2-y1+1}<br><br>"
        
        for i, color in enumerate(['蓝色', '绿色', '红色']):
            channel = roi[:,:,i]
            stats_text += f"<b>{color}通道:</b><br>"
            stats_text += f"均值: {np.mean(channel):.2f}<br>"
            stats_text += f"标准差: {np.std(channel):.2f}<br>"
            stats_text += f"最小值: {np.min(channel)}<br>"
            stats_text += f"最大值: {np.max(channel)}<br><br>"
        
        stats_label.setText(stats_text) 
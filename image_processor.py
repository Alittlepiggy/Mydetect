import cv2
import numpy as np
from scipy import fftpack
import os
import sys

# 添加YOLOv12的导入
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("警告: 未安装ultralytics库，AI检测功能将不可用")

class ImageProcessor:
    def __init__(self):
        self.original_image = None
        self.current_image = None
        # 添加参数存储
        self.brightness = 0
        self.contrast = 1.0
        self.fft_radius = 30
        self.morph_size = 3
        self.canny_low = 50
        self.canny_high = 150
        
        # 添加YOLOv12相关属性
        self.yolo_model = None
        self.yolo_classes = ['pothole']  # 默认类别
        self.yolo_colors = {'pothole': (0, 255, 0), 'default': (0, 0, 255)}
        self.yolo_confidence = 0.3
        self.yolo_iou = 0.45
        
    def load_image(self, image_path):
        """加载图片并进行错误处理"""
        try:
            # 处理中文路径
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}\n请确保文件格式正确且未损坏")
            self.original_image = img
            self.current_image = self.original_image.copy()
            return self.original_image
        except Exception as e:
            raise ValueError(f"加载图片失败: {image_path}\n错误信息: {str(e)}")
        
    def reset_image(self):
        self.current_image = self.original_image.copy()
        
    def adjust_brightness_contrast(self, brightness=0, contrast=1):
        """调整亮度和对比度
        brightness: -100 到 100
        contrast: 0 到 3
        """
        # 每次调节都基于原始图像
        image = self.original_image.astype(np.float32)
        
        # 对比度调节
        if contrast != 1:
            mean = np.mean(image)
            adjusted = (image - mean) * contrast + mean
        else:
            adjusted = image
        
        # 亮度调节
        if brightness != 0:
            adjusted = adjusted + brightness
        
        # 确保值在0-255范围内
        adjusted = np.clip(adjusted, 0, 255)
        
        # 转换回uint8类型
        self.current_image = adjusted.astype(np.uint8)
        return self.current_image

    def clahe_enhancement(self, clip_limit=2.0, tile_size=8):
        """CLAHE自适应直方图均衡化"""
        lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        self.current_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self.current_image

    def fft_filter(self):
        """FFT高通滤波"""
        # 使用当前图像进行处理，而不是原图
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        f = fftpack.fft2(gray)
        fshift = fftpack.fftshift(f)
        
        rows, cols = gray.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        mask[crow-self.fft_radius:crow+self.fft_radius, 
             ccol-self.fft_radius:ccol+self.fft_radius] = 0
        
        fshift = fshift * mask
        f_ishift = fftpack.ifftshift(fshift)
        img_back = fftpack.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        img_back = ((img_back - img_back.min()) * 255 / 
                   (img_back.max() - img_back.min())).astype(np.uint8)
        return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)

    def detect_edges(self):
        """Canny边缘检测"""
        # 使用当前图像进行处理，而不是原图
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def enhance_image(self):
        """综合图像增强处理"""
        try:
            # 转换到LAB颜色空间
            lab = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE处理
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # 高斯滤波去噪
            l = cv2.GaussianBlur(l, (3,3), 0)
            
            # 合并通道
            enhanced_lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # 锐化处理
            kernel = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            # 确保值在合法范围内
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            self.current_image = enhanced
            return self.current_image
        except Exception as e:
            print(f"图像增强处理出错: {str(e)}")
            return self.current_image

    def detect_cracks(self):
        """检测裂缝并返回边界框"""
        # 预处理
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # 自适应阈值分割
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(
            opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 绘制边界框
        result_image = self.current_image.copy()
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(cnt)
                boxes.append((x, y, w, h))
                cv2.rectangle(result_image, (x, y), 
                            (x+w, y+h), (0, 255, 0), 2)
        
        self.current_image = result_image
        return self.current_image, boxes

    def detect_cracks_advanced(self):
        """高级裂缝检测算法"""
        try:
            # 预处理
            gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
            
            # 自适应直方图均衡化
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray = clahe.apply(gray)
            
            # 高斯滤波去噪
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 形态学梯度
            kernel = np.ones((3,3), np.uint8)
            gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)
            
            # Otsu自适应阈值分割
            _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = np.ones((3,3), np.uint8)
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # 查找轮廓
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 分析和绘制结果
            result_image = self.current_image.copy()
            boxes = []
            
            for cnt in contours:
                try:
                    area = cv2.contourArea(cnt)
                    if area > 50:  # 面积阈值
                        # 计算最小外接矩形
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = box.astype(np.int32)
                        
                        # 计算矩形的长宽比
                        width = rect[1][0]
                        height = rect[1][1]
                        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                        
                        # 根据形状特征筛选
                        if aspect_ratio > 2:  # 裂缝通常细长
                            x, y, w, h = cv2.boundingRect(cnt)
                            boxes.append((x, y, w, h))
                            # 绘制轮廓和边界框
                            cv2.drawContours(result_image, [box], 0, (0, 255, 0), 2)
                            cv2.rectangle(result_image, (x,y), (x+w,y+h), (255, 0, 0), 2)
                except Exception as e:
                    print(f"处理轮廓时出错: {str(e)}")
                    continue
            
            self.current_image = result_image
            return self.current_image, boxes
        except Exception as e:
            print(f"缺陷检测出错: {str(e)}")
            return self.current_image, []

    def detect_defects_intelligent(self):
        """智能路面缺陷检测算法 - 自适应增强版"""
        try:
            # 1. 预处理和自适应参数计算
            img = self.current_image.copy()
            
            # 1.1 多尺度处理
            img_small = cv2.resize(img, None, fx=0.5, fy=0.5)
            img_large = cv2.resize(img, None, fx=1.5, fy=1.5)
            images = [img_small, img, img_large]
            
            # 1.2 颜色空间转换和统计特征计算
            features = []
            stats = []
            for scale_img in images:
                # 转换颜色空间
                gray = cv2.cvtColor(scale_img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(scale_img, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(scale_img, cv2.COLOR_BGR2LAB)
                
                # 计算统计特征
                mean_brightness = np.mean(gray)
                std_brightness = np.std(gray)
                global_contrast = (np.percentile(gray, 95) - np.percentile(gray, 5)) / 255.0
                
                features.append((gray, hsv, lab))
                stats.append((mean_brightness, std_brightness, global_contrast))
            
            # 2. 多尺度特征提取
            defect_candidates = {'cracks': [], 'potholes': [], 'water': []}
            
            for idx, ((gray, hsv, lab), (mean_brightness, std_brightness, global_contrast)) in enumerate(zip(features, stats)):
                scale_factor = 0.5 if idx == 0 else (1.5 if idx == 2 else 1.0)
                
                # 2.1 自适应CLAHE增强
                clip_limit = max(2.0, min(4.0, 3.0 * (1 - global_contrast)))
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
                l_enhanced = clahe.apply(lab[:,:,0])
                
                # 2.2 自适应梯度特征
                ksize = 3 if global_contrast > 0.4 else 5
                gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
                gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
                gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
                gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # 3. 缺陷检测
                # 3.1 裂缝检测
                crack_thresh = np.mean(gradient_mag) + 1.5 * np.std(gradient_mag)
                _, crack_binary = cv2.threshold(gradient_mag, crack_thresh, 255, cv2.THRESH_BINARY)
                
                # 自适应形态学处理
                crack_kernel_size = max(3, min(7, int(gray.shape[0] * 0.005)))
                if crack_kernel_size % 2 == 0:
                    crack_kernel_size += 1
                crack_kernel = np.ones((crack_kernel_size, crack_kernel_size), np.uint8)
                crack_mask = cv2.morphologyEx(crack_binary, cv2.MORPH_CLOSE, crack_kernel)
                
                crack_contours, _ = cv2.findContours(crack_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in crack_contours:
                    area = cv2.contourArea(cnt)
                    min_crack_area = gray.shape[0] * gray.shape[1] * 0.0001
                    if area < min_crack_area:
                        continue
                    
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)  
                    
                    width = rect[1][0]
                    height = rect[1][1]
                    aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                    aspect_thresh = 2.5 if global_contrast > 0.5 else 2.0
                    
                    if aspect_ratio > aspect_thresh:
                        x, y, w, h = cv2.boundingRect(cnt)
                        # 调整坐标到原始图像尺寸
                        x, y = int(x/scale_factor), int(y/scale_factor)
                        w, h = int(w/scale_factor), int(h/scale_factor)
                        defect_candidates['cracks'].append((x, y, w, h))
                
                # 3.2 坑洼检测
                block_size = int(min(gray.shape) * 0.02) // 2 * 2 + 1
                block_size = max(3, min(block_size, 21))  # 确保block_size为奇数且在合理范围内
                c_value = max(5, min(15, int(std_brightness * 0.3)))

                pothole_binary = cv2.adaptiveThreshold(
                    l_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, block_size, c_value
                )
                
                pothole_kernel_size = max(3, min(5, int(gray.shape[0] * 0.01)))
                if pothole_kernel_size % 2 == 0:
                    pothole_kernel_size += 1
                pothole_kernel = np.ones((pothole_kernel_size, pothole_kernel_size), np.uint8)
                pothole_mask = cv2.morphologyEx(pothole_binary, cv2.MORPH_OPEN, pothole_kernel)
                pothole_mask = cv2.morphologyEx(pothole_mask, cv2.MORPH_CLOSE, pothole_kernel*2, iterations=7)
                pothole_contours, _ = cv2.findContours(pothole_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in pothole_contours:
                    area = cv2.contourArea(cnt)
                    min_pothole_area = gray.shape[0] * gray.shape[1] * 0.005
                    if area < min_pothole_area:
                        continue
                    
                    x, y, w, h = cv2.boundingRect(cnt)
                    # 调整坐标到原始图像尺寸
                    x, y = int(x/scale_factor), int(y/scale_factor)
                    w, h = int(w/scale_factor), int(h/scale_factor)
                    defect_candidates['potholes'].append((x, y, w, h))
                
                # 3.3 积水检测
                h, s, v = cv2.split(hsv)
                mean_v = np.mean(v)
                mean_s = np.mean(s)
                
                v_thresh = mean_v + std_brightness * 0.5
                s_thresh = mean_s * 0.5
                
                water_mask = cv2.inRange(hsv, (0, 0, v_thresh), (180, s_thresh, 255))
                
                water_kernel_size = max(7, min(15, int(gray.shape[0] * 0.015)))
                if water_kernel_size % 2 == 0:
                    water_kernel_size += 1
                water_kernel = np.ones((water_kernel_size, water_kernel_size), np.uint8)
                water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, water_kernel, iterations=1)
                # cv2.imshow('water_mask', water_mask)
                # cv2.waitKey(0)

                water_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in water_contours:
                    area = cv2.contourArea(cnt)
                    min_water_area = gray.shape[0] * gray.shape[1] * 0.005
                    max_water_area = gray.shape[0] * gray.shape[1] * 0.1
                    if area < min_water_area:
                        continue
                    if area > max_water_area:
                        continue

                    x, y, w, h = cv2.boundingRect(cnt)
                    # 调整坐标到原始图像尺寸
                    x, y = int(x/scale_factor), int(y/scale_factor)
                    w, h = int(w/scale_factor), int(h/scale_factor)
                    defect_candidates['water'].append((x, y, w, h))
            
            # 4. 非极大值抑制和结果融合
            result_image = self.current_image.copy()
            defects = {'cracks': [], 'potholes': [], 'water': []}
            
            for defect_type in defect_candidates:
                boxes = np.array(defect_candidates[defect_type])
                if len(boxes) == 0:
                    continue
                    
                # 计算boxes的面积
                areas = boxes[:,2] * boxes[:,3]
                
                # 按面积排序
                idxs = np.argsort(areas)[::-1]
                
                while len(idxs) > 0:
                    i = idxs[0]
                    defects[defect_type].append(tuple(boxes[i]))
                    
                    if len(idxs) == 1:
                        break
                    
                    # 计算IoU
                    xx1 = np.maximum(boxes[i,0], boxes[idxs[1:],0])
                    yy1 = np.maximum(boxes[i,1], boxes[idxs[1:],1])
                    xx2 = np.minimum(boxes[i,0] + boxes[i,2], boxes[idxs[1:],0] + boxes[idxs[1:],2])
                    yy2 = np.minimum(boxes[i,1] + boxes[i,3], boxes[idxs[1:],1] + boxes[idxs[1:],3])
                    
                    w = np.maximum(0, xx2 - xx1 + 1)
                    h = np.maximum(0, yy2 - yy1 + 1)
                    
                    overlap = (w * h) / (areas[idxs[1:]] + areas[i] - w * h)
                    
                    # 删除重叠较大的框
                    idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > 0.3)[0] + 1)))
            
            # 5. 绘制结果
            colors = {
                'cracks': (0, 255, 0),
                'potholes': (255, 0, 0),
                'water': (0, 0, 255)
            }
            
            labels = {
                'cracks': 'Crack',
                'potholes': 'Pothole',
                'water': 'Water'
            }
            
            for defect_type, boxes in defects.items():
                for (x, y, w, h) in boxes:
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), colors[defect_type], 2)
                    cv2.putText(result_image, labels[defect_type], (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[defect_type], 2)
            
            return result_image, defects
            
        except Exception as e:
            print(f"智能检测出错: {str(e)}")
            return self.current_image, {'cracks': [], 'potholes': [], 'water': []} 

    def load_yolo_model(self, model_path=None):
        """加载YOLOv12模型"""
        if not YOLO_AVAILABLE:
            raise ImportError("未安装ultralytics库，无法使用YOLOv12功能")
            
        if model_path is None:
            # 使用默认模型路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'yolov12','weights','best.pt')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
        try:
            self.yolo_model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"加载YOLOv12模型失败: {str(e)}")
            return False
            
    def detect_with_yolo(self, image=None, save_path=None):
        """使用YOLOv12进行缺陷检测"""
        if not YOLO_AVAILABLE:
            raise ImportError("未安装ultralytics库，无法使用YOLOv12功能")
            
        if self.yolo_model is None:
            if not self.load_yolo_model():
                raise RuntimeError("YOLOv12模型未加载")
                
        # 使用当前图像或提供的图像
        if image is None:
            if self.current_image is None:
                raise ValueError("没有可处理的图像")
            image = self.current_image.copy()
            
        # 保存临时图像用于YOLO处理
        temp_path = "temp_for_yolo.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            # 推理预测
            results = self.yolo_model.predict(temp_path, conf=self.yolo_confidence, iou=self.yolo_iou)
            
            # 绘制检测结果
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 获取颜色和标签
                class_name = self.yolo_classes[cls_id] if cls_id < len(self.yolo_classes) else "unknown"
                color = self.yolo_colors.get(class_name, self.yolo_colors['default'])
                label = f"{class_name} {conf:.2f}"
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签背景
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
                
                # 添加文本
                cv2.putText(image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # 保存结果
            if save_path:
                cv2.imwrite(save_path, image)
                
            return image, results[0].boxes
            
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def detect_defects_ai(self):
        """使用AI方法进行缺陷检测"""
        if self.current_image is None:
            return self.current_image, {'cracks': [], 'potholes': [], 'water': []}
            
        try:
            # 使用YOLOv12进行检测
            result_image, boxes = self.detect_with_yolo()
            
            # 将YOLO检测结果转换为标准格式，只保留pothole类别
            defects = {'cracks': [], 'potholes': [], 'water': []}
            
            for box in boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 只处理pothole类别
                if cls_id < len(self.yolo_classes):
                    class_name = self.yolo_classes[cls_id]
                    if class_name == 'pothole':
                        defects['potholes'].append((x1, y1, x2-x1, y2-y1))
            
            return result_image, defects
            
        except Exception as e:
            print(f"AI检测出错: {str(e)}")
            return self.current_image, {'cracks': [], 'potholes': [], 'water': []} 

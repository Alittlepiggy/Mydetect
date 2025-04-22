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
        self.segment_model = None  # 添加分割模型
        self.yolo_classes = ['pothole']  # 默认类别
        self.yolo_colors = {'pothole': (0, 0, 255), 'default': (0, 0, 255)}
        self.yolo_confidence = 0.3
        self.yolo_iou = 0.45
        self.detection_mode = 'bbox'  # 新增检测模式：'bbox', 'segment', 'both'
        
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
            
    def detect_with_yolo(self, image=None):
        """使用YOLOv12进行检测"""
        if not YOLO_AVAILABLE:
            raise ImportError("未安装ultralytics库，无法使用YOLOv12功能")
            
        if self.yolo_model is None:
            if not self.load_yolo_model():
                raise RuntimeError("YOLOv12模型未加载")
                
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
            result = results[0]
            
            # 创建结果图像
            result_image = image.copy()
            areas = []
            
            # 处理检测结果
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # 获取边界框坐标
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # 计算面积
                    area = (x2 - x1) * (y2 - y1)
                    areas.append(area)
                    
                    # 获取类别和置信度
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    
                    # 获取标签
                    class_name = self.yolo_classes[cls_id] if cls_id < len(self.yolo_classes) else "unknown"
                    label = f"{class_name} {conf:.2f}"
                    
                    # 使用红色绘制边界框
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # 绘制标签背景
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(result_image, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 255), -1)
                    
                    # 添加白色文本
                    cv2.putText(result_image, label, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            return result_image, result.boxes, areas
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def load_segment_model(self, model_path=None):
        """加载分割模型"""
        if not YOLO_AVAILABLE:
            raise ImportError("未安装ultralytics库，无法使用分割功能")
            
        if model_path is None:
            # 使用默认模型路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, 'segment', 'train3', 'weights', 'best.pt')
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"分割模型文件不存在: {model_path}")
            
        try:
            self.segment_model = YOLO(model_path)
            return True
        except Exception as e:
            print(f"加载分割模型失败: {str(e)}")
            return False

    def detect_with_segment(self, image=None):
        """使用分割模型进行检测"""
        if not YOLO_AVAILABLE:
            raise ImportError("未安装ultralytics库，无法使用分割功能")
            
        if self.segment_model is None:
            if not self.load_segment_model():
                raise RuntimeError("分割模型未加载")
                
        if image is None:
            if self.current_image is None:
                raise ValueError("没有可处理的图像")
            image = self.current_image.copy()
            
        # 保存临时图像用于处理
        temp_path = "temp_for_segment.jpg"
        cv2.imwrite(temp_path, image)
        
        try:
            # 推理预测
            results = self.segment_model.predict(
                temp_path,
                conf=self.yolo_confidence,
                iou=self.yolo_iou,
                imgsz=640
            )
            
            # 获取标注后的图像和掩码面积
            result = results[0]
            mask_areas = []
            
            if hasattr(result, 'masks') and result.masks is not None:
                annotated_image = result.plot()
                # 计算每个掩码的像素数
                for mask in result.masks.data:
                    mask_np = mask.cpu().numpy()
                    area = np.sum(mask_np)  # 计算掩码中为True的像素数
                    mask_areas.append(int(area))
            else:
                annotated_image = image.copy()
            
            return annotated_image, result, mask_areas
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def detect_defects_ai(self):
        """使用AI方法进行缺陷检测"""
        if self.current_image is None:
            return self.current_image, {'cracks': [], 'potholes': [], 'water': [], 'stats': {}}
            
        try:
            result_image = self.current_image.copy()
            defects = {
                'cracks': [], 
                'potholes': [], 
                'water': [], 
                'stats': {
                    'bbox': {'count': 0, 'areas': []},
                    'segment': {'count': 0, 'areas': []}
                }
            }
            
            if self.detection_mode in ['bbox', 'both']:
                # 使用YOLOv12进行边界框检测
                bbox_image, boxes, bbox_areas = self.detect_with_yolo()
                result_image = bbox_image
                
                # 处理边界框结果
                pothole_count = 0
                for box, area in zip(boxes, bbox_areas):
                    cls_id = int(box.cls)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    if cls_id < len(self.yolo_classes):
                        class_name = self.yolo_classes[cls_id]
                        if class_name == 'pothole':
                            defects['potholes'].append((x1, y1, x2-x1, y2-y1))
                            pothole_count += 1
                
                defects['stats']['bbox'] = {
                    'count': pothole_count,
                    'areas': bbox_areas
                }
            
            if self.detection_mode in ['segment', 'both']:
                # 使用分割模型进行检测
                segment_image, segment_results, mask_areas = self.detect_with_segment()
                
                # 更新分割统计信息
                if hasattr(segment_results, 'boxes'):
                    segment_count = len(segment_results.boxes)
                else:
                    segment_count = 0
                
                defects['stats']['segment'] = {
                    'count': segment_count,
                    'areas': mask_areas
                }
                
                if self.detection_mode == 'segment':
                    result_image = segment_image
                elif self.detection_mode == 'both':
                    # 确保图像大小一致
                    if segment_image.shape != result_image.shape:
                        segment_image = cv2.resize(segment_image, (result_image.shape[1], result_image.shape[0]))
                    alpha = 0.5
                    result_image = cv2.addWeighted(result_image, 1-alpha, segment_image, alpha, 0)
            
            return result_image, defects
            
        except Exception as e:
            print(f"AI检测出错: {str(e)}")
            return self.current_image, {'cracks': [], 'potholes': [], 'water': [], 'stats': {}} 
        
    def connect_edges(self, edges, min_threshold=5, max_threshold=15):
        """优化版边缘连接算法，使用网格空间分区加速，支持阈值范围"""
        # 使用类属性作为默认值
        min_threshold = min_threshold if min_threshold is not None else self.connect_threshold
        max_threshold = max_threshold if max_threshold is not None else self.connect_max_threshold
        
        # 获取轮廓（兼容OpenCV 3.4.2）
        if cv2.__version__.startswith('3'):
            _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) < 2:
            return edges
        
        # 收集端点并构建空间网格
        endpoints = []
        grid = {}
        grid_size = max(max_threshold, 5)  # 网格大小为最大阈值
        
        for cnt in contours:
            if len(cnt) >= 2:
                pt_start = tuple(cnt[0][0])
                pt_end = tuple(cnt[-1][0])
                endpoints.extend([pt_start, pt_end])
        
        # 构建网格空间索引
        for idx, (x, y) in enumerate(endpoints):
            grid_x = x // grid_size
            grid_y = y // grid_size
            if (grid_x, grid_y) not in grid:
                grid[(grid_x, grid_y)] = []
            grid[(grid_x, grid_y)].append(idx)
        
        # 创建输出图像
        connected = np.zeros_like(edges)
        cv2.drawContours(connected, contours, -1, 255, 1)
        
        # 遍历所有端点，仅在相邻网格中查找邻近点
        for i, (x, y) in enumerate(endpoints):
            current_grid = (x // grid_size, y // grid_size)
            
            # 检查当前网格和8个相邻网格
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor_grid = (current_grid[0] + dx, current_grid[1] + dy)
                    if neighbor_grid not in grid:
                        continue
                    
                    # 检查该网格内的所有点
                    for j in grid[neighbor_grid]:
                        if j <= i:  # 避免重复检查
                            continue
                        
                        # 计算距离
                        x2, y2 = endpoints[j]
                        distance = np.sqrt((x - x2)**2 + (y - y2)**2)
                        # 如果距离在阈值范围内，则连接
                        if min_threshold <= distance <= max_threshold:
                            cv2.line(connected, (x, y), (x2, y2), 255, 1)
        
        return connected


    def defectdetect_matlab(self):
        """
        使用传统方法进行缺陷检测。
        :return: 检测结果图像和缺陷区域框的列表。
        """
        # 预处理
        img = self.current_image.copy()
        gray = self.preprocess_image(img)
        ref_block = self.getRef(gray)

        if ref_block is None:
            return img, []

        # 滑动窗口参数
        window_size = 3
        stride = 1

        # 存储检测到的缺陷区域
        defect_boxes = []
        height, width = gray.shape

        # 滑动窗口检测
        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                # 提取当前窗口
                window = gray[y:y + window_size, x:x + window_size]

                # 计算相似度
                sim = self.getsim(ref_block, window)

                # 如果相似度低于阈值，认为是缺陷
                if sim < 0.9:  # 可调整阈值
                    defect_boxes.append([x, y, x + window_size, y + window_size])

        # 转换为 NumPy 数组
        if len(defect_boxes) > 0:
            defect_boxes = np.array(defect_boxes)

            # 合并重叠的框
            defect_boxes = self.merge_boxes(defect_boxes)

            # 过滤不合适大小的框
            defect_boxes = self.filter_boxes(defect_boxes)

            # 在原图上标记缺陷
            result_img = img.copy()
            for box in defect_boxes:
                cv2.rectangle(result_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        else:
            defect_boxes = np.array([])
            result_img = img.copy()

        return result_img, defect_boxes

    def preprocess_image(self, img):
        """
        图像预处理：灰度化和归一化。
        :param img: 输入图像（BGR 格式）。
        :return: 归一化的灰度图像。
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_double = gray.astype(np.float32) / 255.0
        return gray_double

    def getBad(self, img):
        """
        获取图像中的坏点图。
        :param img: 输入灰度图像。
        :return: 坏点图（二值图像）。
        """
        h, w = img.shape
        win_len = 10
        scale = np.zeros((h, w))

        for i in range(0, h - win_len + 1, win_len):
            for j in range(0, w - win_len + 1, win_len):
                region = img[i:i + win_len, j:j + win_len]
                mx = np.max(region)
                mn = np.min(region)
                # 避免除零，同时确保计算结果有意义
                if mn == 0:
                    mn = 1e-6  # 使用极小值代替零
                scale[i:i + win_len, j:j + win_len] = (mx * 3.0) / float(mn)

        # 归一化处理
        scale = cv2.normalize(scale, None, 0, 1, cv2.NORM_MINMAX)

        # 确保所有值为非负数
        scale = np.clip(scale, 0, None)

        gamma = 0.5
        J = np.power(scale, gamma)

        # 使用均值作为阈值生成二值图像
        return (J > np.mean(J)).astype(np.uint8)

    def getRef(self, pic):
        """
        获取参考块。
        :param pic: 输入灰度图像。
        :return: 参考块。
        """
        bad = self.getBad(pic)
        h, w = pic.shape
        ref_block = np.zeros((3, 3))
        cnt = 0

        for i in range(0, h - 3, 3):
            for j in range(0, w - 3, 3):
                tmp = bad[i:i + 3, j:j + 3]
                if np.min(tmp) == 1:
                    ref_block += self.normalize(pic[i:i + 3, j:j + 3])
                    cnt += 1

        if cnt > 0:
            ref_block /= cnt
        return ref_block

    def normalize(self, x):
        """
        归一化函数。
        :param x: 输入数组。
        :return: 归一化后的数组。
        """
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm

    def getsim(self, ref, x):
        """
        计算相似度。
        :param ref: 参考块。
        :param x: 当前窗口。
        :return: 最大相似度值。
        """
        href, wref = ref.shape
        hx, wx = x.shape
        hout = href - hx + 1
        wout = wref - wx + 1
        steph = hx // 2
        stepw = wx // 2
        S = np.zeros((hout, wout))
        for i in range(0, hout, steph):
            for j in range(0, wout, stepw):
                slice_ref = ref[i:i + hx, j:j + wx]
                S[i, j] = np.sum(self.normalize(slice_ref) * self.normalize(x))
        return np.max(S)

    def merge_boxes(self, boxes):
        """
        合并重叠的矩形框。
        :param boxes: 输入的矩形框列表。
        :return: 合并后的矩形框列表。
        """
        if len(boxes) == 0:
            return []

        merged = []
        while len(boxes) > 0:
            r = boxes[0]
            boxes = boxes[1:]
            overlap = True

            while overlap:
                overlap_idx = self.findRectOverlap(r, boxes)
                if len(overlap_idx) == 0:
                    overlap = False
                else:
                    # 合并所有重叠的
                    r[0] = min([r[0]] + [boxes[i][0] for i in overlap_idx])
                    r[1] = min([r[1]] + [boxes[i][1] for i in overlap_idx])
                    r[2] = max([r[2]] + [boxes[i][2] for i in overlap_idx])
                    r[3] = max([r[3]] + [boxes[i][3] for i in overlap_idx])
                    boxes = np.delete(boxes, overlap_idx, axis=0)

            merged.append(r)

        return np.array(merged)

    def findRectOverlap(self, r, rects):
        """
        判断哪些矩形与 r 有交集。
        :param r: 当前矩形框。
        :param rects: 其他矩形框列表。
        :return: 重叠矩形框的索引列表。
        """
        idx = []
        for i, rect in enumerate(rects):
            if not (rect[0] > r[2] or rect[2] < r[0] or rect[1] > r[3] or rect[3] < r[1]):
                idx.append(i)
        return idx

    def filter_boxes(self, boxes, min_area=100):
        """
        过滤太小的矩形框。
        :param boxes: 输入的矩形框列表。
        :param min_area: 最小面积阈值。
        :return: 过滤后的矩形框列表。
        """
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[areas >= min_area]
import os
import cv2
import numpy as np
from ultralytics import YOLO

class YOLOv12Detector:
    def __init__(self, model_path, classes, colors):
        self.model = YOLO(model_path)
        self.classes = classes
        self.colors = colors  # {'pothole': (0,255,0), ...}

    def detect_and_visualize(self, img_path, output_path):
        # 推理预测
        results = self.model.predict(img_path, conf=0.3, iou=0.45)
        
        # 读取原始图像
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("无法读取图像文件")

        # 绘制检测结果
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 获取颜色和标签
            color = self.colors.get(self.classes[cls_id], (0,0,255))
            label = f"{self.classes[cls_id]} {conf:.2f}"

            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            
            # 添加文本
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        # 保存结果
        cv2.imwrite(output_path, image)

def batch_process_images(detector, input_folder, output_folder):
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 遍历输入文件夹下所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            print(f"Processing {img_path} -> {output_path}")
            try:
                detector.detect_and_visualize(img_path, output_path)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

# 使用示例
if __name__ == "__main__":
    detector = YOLOv12Detector(
        model_path='runs/detect/train/weights/best.pt',
        classes=['pothole'],
        colors={'pothole': (0, 255, 0), 'default': (0,0,255)}
    )
    # 指定要批处理图片的输入和输出文件夹（请根据需要修改路径）
    input_folder = "IMAGES"
    output_folder = "outputs"
    batch_process_images(detector, input_folder, output_folder)
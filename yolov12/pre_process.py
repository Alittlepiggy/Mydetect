import os
import shutil
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def voc_to_yolo(xml_dir, img_src_dir, output_dir, classes, test_size=0.2, seed=42):
    """
    参数说明：
    xml_dir: VOC标注文件目录
    img_src_dir: 原始图片目录
    output_dir: YOLO格式输出根目录
    classes: 类别列表
    test_size: 验证集比例 (默认0.2)
    seed: 随机种子 (默认42)
    """
    # 创建标准YOLO目录结构
    dirs = {
        'images/train': os.path.join(output_dir, 'images/train'),
        'images/val': os.path.join(output_dir, 'images/val'),
        'labels/train': os.path.join(output_dir, 'labels/train'),
        'labels/val': os.path.join(output_dir, 'labels/val')
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 获取所有XML文件（不带扩展名）
    xml_files = [os.path.splitext(f)[0] for f in os.listdir(xml_dir) if f.endswith('.xml')]
    
    # 划分训练集和验证集
    train_names, val_names = train_test_split(xml_files, test_size=test_size, random_state=seed)
    
    # 处理每个划分集
    for split, names in [('train', train_names), ('val', val_names)]:
        for name in names:
            # 1. 处理图片文件（支持jpg/png格式）
            img_ext = None
            for ext in ['.jpg', '.png']:
                src_img = os.path.join(img_src_dir, name + ext)
                if os.path.exists(src_img):
                    img_ext = ext
                    break
            if not img_ext:
                print(f"警告：未找到图片 {name}")
                continue
            
            # 复制图片到目标目录
            dest_img_dir = dirs[f'images/{split}']
            shutil.copy(src_img, os.path.join(dest_img_dir, name + img_ext))
            
            # 2. 转换并保存标签
            src_xml = os.path.join(xml_dir, name + '.xml')
            dest_label_dir = dirs[f'labels/{split}']
            
            # 解析XML
            tree = ET.parse(src_xml)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 生成YOLO格式标签
            yolo_lines = []
            for obj in root.findall('object'):
                cls_name = obj.find('name').text.strip()
                if cls_name not in classes:
                    continue
                cls_id = classes.index(cls_name)
                
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # 转换为YOLO格式
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
            
            # 写入标签文件
            with open(os.path.join(dest_label_dir, name + '.txt'), 'w') as f:
                f.write('\n'.join(yolo_lines))

# 使用示例
voc_to_yolo(
    xml_dir='ANNOTATIONS',      # VOC标注目录
    img_src_dir='IMAGES',       # 原始图片目录
    output_dir='datasets',  # 输出目录
    classes=['pothole'],        # 类别列表
    test_size=0.2
)
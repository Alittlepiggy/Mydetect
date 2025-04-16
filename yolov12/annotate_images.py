import os
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm  # 进度条支持

def draw_annotations(images_dir, xmls_dir, output_dir):
    """
    参数说明：
    images_dir: 原始图片目录（需与XML文件名对应）
    xmls_dir: XML标注文件目录
    output_dir: 标注结果输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取匹配的文件列表
    img_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))])
    xml_files = sorted([f for f in os.listdir(xmls_dir) if f.endswith('.xml')])
    
    # 颜色配置（可扩展多类别）
    color_mapping = {
        'pothole': (0, 255, 0),   # BGR格式-绿色
        'default': (0, 0, 255)    # 红色作为默认
    }

    # 处理每张图片
    for img_file, xml_file in tqdm(zip(img_files, xml_files), total=len(img_files)):
        # 1. 读取图片
        img_path = os.path.join(images_dir, img_file)
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告：无法读取图片 {img_path}")
            continue

        # 2. 解析XML
        xml_path = os.path.join(xmls_dir, xml_file)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as e:
            print(f"XML解析失败：{xml_path} - {str(e)}")
            continue

        # 3. 遍历所有标注对象
        for obj in root.findall('object'):
            # 获取类别名称
            name = obj.find('name').text.strip().lower()
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            # 4. 绘制检测框
            color = color_mapping.get(name, color_mapping['default'])
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            
            # 5. 添加类别标签
            label = f"{name}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (xmin, ymin - text_height - 4), (xmin + text_width, ymin), color, -1)
            cv2.putText(image, label, (xmin, ymin - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 6. 保存结果
        output_path = os.path.join(output_dir, f"annotated_{img_file}")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # 路径配置
    IMAGES_DIR = "IMAGES"
    XMLS_DIR = "ANNOTATIONS"
    OUTPUT_DIR = "ANNOTATED_IMAGES"
    
    draw_annotations(IMAGES_DIR, XMLS_DIR, OUTPUT_DIR)
    print(f"标注完成！结果已保存至 {OUTPUT_DIR}")
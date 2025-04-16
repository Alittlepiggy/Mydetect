import os 

def check_dataset(data_dir):
    """ 验证图片与标签的对应关系 """
    from PIL import Image
    
    for split in ['train', 'val']:
        img_dir = os.path.join(data_dir, 'images', split)
        label_dir = os.path.join(data_dir, 'labels', split)
        
        print(f"\n检查{split}集:")
        for img_file in os.listdir(img_dir):
            # 验证图片可读性
            img_path = os.path.join(img_dir, img_file)
            try:
                Image.open(img_path).verify()
            except:
                print(f"损坏图片: {img_path}")
            
            # 验证标签存在性
            label_file = os.path.splitext(img_file)[0] + '.txt'
            if not os.path.exists(os.path.join(label_dir, label_file)):
                print(f"缺失标签: {img_file}")

check_dataset('datasets')
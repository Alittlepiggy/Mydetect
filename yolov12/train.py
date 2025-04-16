from ultralytics import YOLO
import torch

if __name__ == '__main__': 
    # 初始化模型
    model = YOLO('yolov12s.pt')  # 加载预训练权重[8](@ref)
    model.add_callback('on_train_start', lambda trainer: torch.cuda.empty_cache())

    # 训练配置
    config = {
        'data': 'data.yaml',        # 数据集配置文件
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'optimizer': 'AdamW',       # 使用改进优化器
        'lr0': 0.001,              # 初始学习率
        'cos_lr': True,            # 余弦退火策略
        'label_smoothing': 0.1,    # 标签平滑
        'mosaic': 0.5,             # 数据增强概率
        'mixup': 0.2,              # MixUp增强
        'patience': 15             # 早停机制
    }

    # 启动训练
    results = model.train(**config)
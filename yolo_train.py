from ultralytics import YOLO
import torch
import os

def get_train_image_count(dataset_path):
    """
    计算YOLO数据集中训练集的图像数量
    参数:
        dataset_path: YOLO数据集根目录路径，例如"yolo_dataset"
    返回:
        训练集图像数量，如果目录不存在则返回0
    """
    # 构建训练集图像目录路径
    train_images_path = os.path.join(dataset_path, "train", "images")
    # 检查目录是否存在
    if not os.path.exists(train_images_path):
        print(f"警告: 训练集图像目录不存在 - {train_images_path}")
        return 0
    # 定义常见的图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')
    # 统计图像文件数量
    image_count = 0
    for file in os.listdir(train_images_path):
        if file.lower().endswith(image_extensions):
            image_count += 1
    return image_count

if __name__ == '__main__':  # 添加主模块保护
    # 检查GPU是否可用
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")

# 加载模型（选择预训练权重或模型配置文件）
# 预训练模型示例：yolov8n.pt (n/s/m/l/x), yolov9-c.pt等
# 自定义模型：可传入.yaml配置文件，如yolov8n.yaml（会从头训练）
    model = YOLO('yolov8s.pt')
    dataset_size = get_train_image_count("yolo_dataset")
    print(f"训练集图像数量: {dataset_size}")
    cache_strategy = True if dataset_size < 500 else False
    # cache_strategy = False
    print(f"数据集大小: {dataset_size}张图片，启用缓存: {cache_strategy}")

# 训练参数配置
    train_params = {
        'data': './yolo_dataset/custom_data.yaml',  # 数据配置文件路径
        'epochs': 50,  # 训练轮次
        'batch': 8,  # 批次大小（根据GPU显存调整）
        'imgsz': 640,  # 输入图片尺寸
        'device': device,  # 训练设备
        'project': './runs2/train',  # 结果保存主目录
        'name': 'exp',  # 实验名称（用于区分不同训练）
        'cache': cache_strategy,
        'workers': 4,  # 调整数据加载进程数
        'lr0': 0.01,  # 初始学习率
        'weight_decay': 0.0005,  # 权重衰减（防止过拟合）
        'augment': True,  # 启用训练时的数据增强
        'verbose': True,  # 显示详细训练日志
        'save': True,  # 保存模型权重
        'save_period': 5,  # 每x轮保存一次中间权重
        'val': True  # 训练中同时进行验证
    }

    # 启动训练
    results = model.train(**train_params)

    # 训练完成后，在验证集上评估最佳模型
    print("\n===== 开始验证最佳模型 =====")
    metrics = model.val(
        data=train_params['data'],
        device=device,
        imgsz=train_params['imgsz']
    )

    # 打印关键评估指标
    print("\n验证集关键指标：")
    print(f"mAP50-95: {metrics.box.map:.4f}")  # 所有类别的平均精度
    print(f"mAP50: {metrics.box.map50:.4f}")  # IoU=0.5时的平均精度
    print(f"mAP75: {metrics.box.map75:.4f}")  # IoU=0.75时的平均精度
    print(f"平均召回率: {metrics.box.mr:.4f}")  # 平均召回率

import os
import json
import shutil
from sklearn.model_selection import train_test_split

# 配置路径
raw_img_dir = "./raw_dataset/images"  # 原始图片目录
raw_ann_dir = "./raw_dataset/labels"  # LabelMe生成的JSON目录
output_dir = "./yolo_dataset"  # 转换后输出目录
classes_file = "./classes.txt"  # 类别文件

# 创建输出目录
os.makedirs(f"{output_dir}/train/images", exist_ok=True)
os.makedirs(f"{output_dir}/train/labels", exist_ok=True)
os.makedirs(f"{output_dir}/val/images", exist_ok=True)
os.makedirs(f"{output_dir}/val/labels", exist_ok=True)

# 读取类别列表，生成类别到ID的映射
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
class2id = {cls: i for i, cls in enumerate(classes)}

# 获取所有图片路径
img_files = [f for f in os.listdir(raw_img_dir) if f.endswith(('.jpg', '.png'))]
print(f"找到{len(img_files)}张图片")

# 划分训练集和验证集（8:2）
train_imgs, val_imgs = train_test_split(img_files, test_size=0.2, random_state=42)


def convert(imgs, split):
    """转换指定划分的图片和标注"""
    for img_name in imgs:
        # 图片路径
        img_path = os.path.join(raw_img_dir, img_name)
        # JSON标注文件路径（替换扩展名）
        ann_name = os.path.splitext(img_name)[0] + ".json"
        ann_path = os.path.join(raw_ann_dir, ann_name)

        # 复制图片到输出目录
        shutil.copy(img_path, f"{output_dir}/{split}/images/{img_name}")

        # 处理标注文件（若存在）
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                data = json.load(f)

            img_w = data["imageWidth"]  # 图片宽度
            img_h = data["imageHeight"]  # 图片高度
            yolo_labels = []

            for shape in data["shapes"]:
                # 检查类别是否在预定义列表中
                cls = shape["label"]
                if cls not in class2id:
                    print(f"警告：类别'{cls}'不在classes.txt中，已跳过")
                    continue

                # 获取边界框像素坐标（左上角和右下角）
                x1, y1 = shape["points"][0]
                x2, y2 = shape["points"][1]

                # 转换为YOLO格式（归一化）
                x_center = (x1 + x2) / 2 / img_w
                y_center = (y1 + y2) / 2 / img_h
                width = (x2 - x1) / img_w
                height = (y2 - y1) / img_h

                # 确保坐标在0~1范围内
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))

                # 添加到标签列表（类别ID + 坐标）
                yolo_labels.append(f"{class2id[cls]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # 保存YOLO格式标签
            label_name = os.path.splitext(img_name)[0] + ".txt"
            with open(f"{output_dir}/{split}/labels/{label_name}", 'w') as f:
                f.write('\n'.join(yolo_labels))
        else:
            # 无标注文件（无目标），创建空txt
            label_name = os.path.splitext(img_name)[0] + ".txt"
            open(f"{output_dir}/{split}/labels/{label_name}", 'w').close()


# 转换训练集和验证集
print("开始转换训练集...")
convert(train_imgs, "train")
print("开始转换验证集...")
convert(val_imgs, "val")

# 生成YOLO数据配置文件
with open(f"{output_dir}/custom_data.yaml", 'w') as f:
    f.write(f"""train: ./train/images
val: ./val/images
nc: {len(classes)}
names: {classes}
""")

print("转换完成！输出目录：", output_dir)

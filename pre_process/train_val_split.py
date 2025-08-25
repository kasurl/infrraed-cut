import os
import random
import argparse


def split_images(folder_path, val_count, image_extensions=['.jpg', '.jpeg', '.png', '.bmp', '.gif']):
    """
    将文件夹中的图像文件随机划分为训练集和验证集

    参数:
        folder_path: 图像文件所在的文件夹路径
        val_count: 验证集的图像数量
        image_extensions: 被视为图像文件的扩展名列表
    """
    # 获取文件夹中所有图像文件
    image_files = []
    for file in os.listdir(folder_path):
        # 检查文件扩展名是否在图像扩展名列表中
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    # 检查文件数量是否足够
    if len(image_files) < val_count:
        raise ValueError(f"图像文件总数({len(image_files)})少于要求的验证集数量({val_count})")

    # 随机打乱文件列表
    random.shuffle(image_files)

    # 划分训练集和验证集
    val_files = image_files[:val_count]
    train_files = image_files[val_count:]

    # 生成训练集文件列表
    with open('train.txt', 'w', encoding='utf-8') as f:
        for file in train_files:
            f.write(f"{file}\n")

    # 生成验证集文件列表
    with open('val.txt', 'w', encoding='utf-8') as f:
        for file in val_files:
            f.write(f"{file}\n")

    print(f"划分完成！")
    print(f"总图像数量: {len(image_files)}")
    print(f"训练集数量: {len(train_files)}，已保存到 train.txt")
    print(f"验证集数量: {len(val_files)}，已保存到 val.txt")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='将图像文件随机划分为训练集和验证集')
    parser.add_argument('--folder', required=True, help='图像文件所在的文件夹路径')
    parser.add_argument('--val_count', type=int, required=True, help='验证集的图像数量')
    args = parser.parse_args()


    # 调用函数进行划分#   python pre_process/train_val_split.py --folder=/path/to/images --val_count=20%
    split_images(args.folder, args.val_count)
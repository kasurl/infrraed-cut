import os
import shutil
import argparse


def classify_dataset(txt_folder, source_folder, target_folder):
    """
    根据train.txt和val.txt文件将源文件夹中的文件分类到目标文件夹的train和val子文件夹中

    参数:
        txt_folder: 存放train.txt和val.txt的文件夹路径
        source_folder: 需要分类的数据集所在文件夹
        target_folder: 分类后数据存放的目标文件夹
    """
    # 检查必要的文件是否存在
    train_txt_path = os.path.join(txt_folder, 'train.txt')
    val_txt_path = os.path.join(txt_folder, 'val.txt')

    if not os.path.exists(train_txt_path):
        raise FileNotFoundError(f"找不到训练集文件列表: {train_txt_path}")
    if not os.path.exists(val_txt_path):
        raise FileNotFoundError(f"找不到验证集文件列表: {val_txt_path}")

    # 创建目标文件夹
    train_target = os.path.join(target_folder, 'train')
    val_target = os.path.join(target_folder, 'val')

    os.makedirs(train_target, exist_ok=True)
    os.makedirs(val_target, exist_ok=True)

    # 读取train.txt并移动文件
    with open(train_txt_path, 'r', encoding='utf-8') as f:
        train_files = [line.strip() for line in f if line.strip()]

    moved_train = 0
    for file in train_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(train_target, file)

        if os.path.exists(src):
            shutil.copy2(src, dst)  # 使用copy2保留文件元数据
            moved_train += 1
        else:
            print(f"警告: 未找到文件 {src}，已跳过")

    # 读取val.txt并移动文件
    with open(val_txt_path, 'r', encoding='utf-8') as f:
        val_files = [line.strip() for line in f if line.strip()]

    moved_val = 0
    for file in val_files:
        src = os.path.join(source_folder, file)
        dst = os.path.join(val_target, file)

        if os.path.exists(src):
            shutil.copy2(src, dst)
            moved_val += 1
        else:
            print(f"警告: 未找到文件 {src}，已跳过")

    print(f"分类完成！")
    print(f"成功复制 {moved_train} 个文件到训练集文件夹")
    print(f"成功复制 {moved_val} 个文件到验证集文件夹")
    print(f"训练集文件路径: {train_target}")
    print(f"验证集文件路径: {val_target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='根据train.txt和val.txt文件分类数据集')
    parser.add_argument('--txt_folder', default='.', help='存放train.txt和val.txt的文件夹路径，默认当前目录')
    parser.add_argument('--source', required=True, help='需要分类的数据集所在文件夹')
    parser.add_argument('--target', required=True, help='分类后数据存放的目标文件夹')

#   python pre_process/classify_dataset.py --txt_folder=/path/to/images --source=/path/to/images/train --target=/path/to/images/split
    args = parser.parse_args()
    classify_dataset(args.txt_folder, args.source, args.target)
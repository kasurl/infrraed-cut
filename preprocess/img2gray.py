import os
import cv2
from pathlib import Path


def binarize_images(input_dir, output_dir):
    """
    读取输入目录下的所有图像，进行二值化处理后保存到输出目录

    参数:
        input_dir: 输入图像所在目录
        output_dir: 处理后图像的保存目录
        threshold: 二值化阈值，默认127
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 支持的图像文件扩展名
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        # 检查文件是否为图像
        if filename.lower().endswith(image_extensions):
            input_path = os.path.join(input_dir, filename)

            # 读取图像（使用OpenCV）
            try:
                # 以灰度模式读取图像
                image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

                if image is None:
                    print(f"无法读取图像: {input_path}")
                    continue

                # 构建输出路径
                output_path = os.path.join(output_dir, filename)

                # 保存处理后的图像
                cv2.imwrite(output_path, image)
                print(f"已处理并保存: {output_path}")

            except Exception as e:
                print(f"处理图像 {input_path} 时出错: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 输入目录（包含要处理的图像）
    input_directory = "dataset/roadcrack/train/lwir"
    # 输出目录（保存二值化后的图像）
    output_directory = "dataset/roadcrack/val/images"

    # 执行二值化处理
    binarize_images(input_directory, output_directory)
    print("图像处理完成！")
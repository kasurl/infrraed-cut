import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import UNet  # 导入模型类

# 定义与训练时一致的转换（需和data_loader.py保持一致）
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 添加训练时使用的Resize
    transforms.ToTensor(),
])


def predict_cracks(model, image_path, save_dir='results'):
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 加载图像
    img = Image.open(image_path).convert("RGB")
    original_size = img.size  # 保存原始尺寸用于后续恢复

    # 应用相同的转换
    if transform:
        img_tensor = transform(img)

    # 添加批次维度并移动到GPU
    img_tensor = img_tensor.unsqueeze(0).cuda()

    # 使用模型进行预测（关闭梯度计算提高效率）
    with torch.no_grad():
        output = model(img_tensor)

    # 后处理输出：阈值分割并转换为numpy数组
    pred_mask = (output > 0.5).float().squeeze().cpu().numpy()

    # 恢复到原始图像尺寸
    pred_mask = cv2.resize(pred_mask, original_size[::-1])  # (w,h)转(h,w)

    # 处理原始图像用于可视化
    img_np = np.array(img)  # 直接从PIL图像转换为numpy数组（保持原始尺寸）

    # 处理掩码用于可视化（转换为RGB以便拼接）
    pred_mask = (pred_mask * 255).astype(np.uint8)
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)

    # 拼接原始图像和预测结果并保存
    result = np.concatenate((img_np, pred_mask), axis=1)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 转换为BGR格式保存

    # 显示结果
    cv2.imshow('Original vs Prediction', cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return save_path  # 返回保存路径方便后续处理


if __name__ == '__main__':
    model_path = 'saved_models/epoch_19_val_loss_0.0272.pth'  # 模型权重文件路径
    image_path = 'dataset/roadcrack/train/images/1014.png'  # 测试图像路径

    # 加载模型
    model = UNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()  # 设置为评估模式

    # 进行预测
    saved_path = predict_cracks(model, image_path)
    print(f"预测结果已保存至: {saved_path}")



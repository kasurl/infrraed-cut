import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import UNet  # 导入模型类

# 定义与训练时一致的转换（需和data_loader.py保持一致）
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def predict_cracks(model, image_path, save_dir='results'):
    # 加载图像
    img = Image.open(image_path).convert("RGB")

    # 应用相同的转换
    if transform:
        img = transform(img)

    # 添加批次维度
    img = img.unsqueeze(0).cuda()

    # 使用模型进行预测
    output = model(img)

    # 后处理输出
    pred_mask = (output > 0.5).float().squeeze().cpu().numpy()

    # 可视化结果
    img_np = np.array(img.squeeze().permute(1, 2, 0).cpu())
    pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
    result = np.concatenate((img_np, pred_mask), axis=1)

    # 显示结果
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite(os.path.join(save_dir, os.path.basename(image_path)), result * 255)


if __name__ == '__main__':
    model_path = 'path/to/your/best.pth'  # 模型权重文件路径
    image_path = 'path/to/your/image.jpg'  # 测试图像路径

    # 加载模型
    model = UNet().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 进行预测
    predict_cracks(model, image_path)
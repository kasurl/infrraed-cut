import torch
from model import UNet  # 导入模型类
from data_loader import test_dataloader  # 导入数据加载器（建议用验证集加载器）
import torch.optim as optim
import torch.nn as nn

# 加载模型
model_path = 'saved_models/epoch_19_val_loss_0.0272.pth'  # 模型权重文件路径
model = UNet().cuda()
model.load_state_dict(torch.load(model_path))


# 加载模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the training images: {100 * correct / total:.2f}%')
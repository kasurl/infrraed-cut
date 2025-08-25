import torch
from model import UNet  # 导入模型类
from data_loader import dataloader  # 导入数据加载器（建议用验证集加载器）
from model import model,optimizer,criterion

# 加载模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels.unsqueeze(1)).sum().item()

    print(f'Accuracy of the network on the training images: {100 * correct / total:.2f}%')
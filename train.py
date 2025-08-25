import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import UNet
from data_loader import train_dataloader, val_dataloader
from train_met import calculate_iou, save_model_if_best
# 实例化模型
model = UNet().cuda()

# 设置损失函数和优化器
MODEL_DIR = "saved_models"
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 确保保存模型的目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

num_epochs = 100

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    running_train_loss = 0.0
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.float())

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * inputs.size(0)

    train_epoch_loss = running_train_loss / len(train_dataloader.dataset)

    # 验证阶段


    # 验证阶段修改
    model.eval()
    running_val_loss = 0.0
    total_iou = 0.0  # 新增IoU指标
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            labels_float = labels.float()

            # 计算损失
            val_loss = criterion(outputs, labels_float)
            running_val_loss += val_loss.item() * inputs.size(0)

            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels_float.numel()
            correct += (predicted == labels_float).sum().item()

            # 计算IoU
            iou = calculate_iou(predicted, labels_float)
            total_iou += iou.item()  # 直接累加IoU

    val_epoch_loss = running_val_loss / len(val_dataloader.dataset)
    val_accuracy = 100 * correct / total
    val_iou = total_iou / len(val_dataloader)  # 平均IoU

    # 保存验证损失最小的三个模型
    save_model_if_best(model, epoch, val_epoch_loss)

    # 打印时增加IoU
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% | Val IoU: {val_iou:.4f}')



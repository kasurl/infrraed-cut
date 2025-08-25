import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import UNet
from data_loader import train_dataloader, val_dataloader


def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return intersection / (union + 1e-8)  # 避免除零


# 实例化模型
model = UNet().cuda()

# 设置损失函数和优化器
criterion = nn.BCELoss()  # 二分类交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 确保保存模型的目录存在
os.makedirs("saved_models", exist_ok=True)

num_epochs = 100
best_val_losses = [float('inf')] * 3  # 存储最佳的三个验证损失
best_model_paths = ["", "", ""]  # 存储对应最佳模型的路径

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

    current_val_loss = val_epoch_loss
    if current_val_loss < best_val_losses[-1]:
        # 保存模型
        model_path = f"saved_models/model_epoch_{epoch + 1}_val_loss_{current_val_loss:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        # 更新最佳损失和路径
        best_val_losses = sorted(best_val_losses + [current_val_loss])[:3]
        best_model_paths = [model_path if loss == current_val_loss else path
                            for loss, path in zip(best_val_losses, best_model_paths + [""])]

    # 打印时增加IoU
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}% | Val IoU: {val_iou:.4f}')
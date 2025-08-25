import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import UNet, model, optimizer, criterion
from data_loader import train_dataloader, val_dataloader

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
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)

            # 计算验证损失
            val_loss = criterion(outputs, labels.float())
            running_val_loss += val_loss.item() * inputs.size(0)

            # 计算准确率
            predicted = (outputs > 0.5).float()
            total += labels.numel()  # 计算总元素数量而非批次大小
            correct += (predicted == labels.unsqueeze(1)).sum().item()

    val_epoch_loss = running_val_loss / len(val_dataloader.dataset)
    val_accuracy = 100 * correct / total

    # 打印本轮结果
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print(f'Train Loss: {train_epoch_loss:.4f} | Val Loss: {val_epoch_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%')

    # 检查是否需要保存当前模型（保持最佳的三个模型）
    current_model_path = f"saved_models/epoch_{epoch + 1}_val_loss_{val_epoch_loss:.4f}.pth"

    # 找到当前损失在最佳损失列表中的位置
    insert_pos = -1
    for i in range(3):
        if val_epoch_loss < best_val_losses[i]:
            insert_pos = i
            break

    # 如果当前模型足够好，插入到最佳列表并保存
    if insert_pos != -1:
        # 删除被挤出前3的模型文件
        for i in range(3):
            if i != insert_pos and os.path.exists(best_model_paths[i]):
                os.remove(best_model_paths[i])
                print(f"Removed model at {best_model_paths[i]}")

        # 移动后面的模型位置
        for i in range(2, insert_pos, -1):
            best_val_losses[i] = best_val_losses[i - 1]
            best_model_paths[i] = best_model_paths[i - 1]

        # 插入当前模型
        best_val_losses[insert_pos] = val_epoch_loss
        best_model_paths[insert_pos] = current_model_path

        # 保存当前模型
        torch.save(model.state_dict(), current_model_path)
        print(f"Saved new best model at position {insert_pos + 1} to {current_model_path}")

# 最终提示最佳三个模型
print("\nTraining complete! Best 3 models:")
for i, (path, loss) in enumerate(zip(best_model_paths, best_val_losses), 1):
    print(f"Rank {i}: {path} with Val Loss: {loss:.4f}")
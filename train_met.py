import os
import torch

MODEL_DIR = "saved_models"
best_models = []  # 保存验证损失最小的三个模型

def calculate_iou(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection
    return intersection / (union + 1e-8)  # 避免除零


def save_model_if_best(model, epoch, current_val_loss):
    """保存模型并只保留验证损失最小的三个模型"""
    global best_models

    # 生成唯一的模型文件名，包含 epoch 和验证损失
    model_filename = f"model_epoch_{epoch + 1}_val_loss_{current_val_loss:.4f}.pth"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # 保存模型到本地
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到: {model_path}")

    # 将当前模型添加到候选列表
    candidates = best_models + [(current_val_loss, model_path)]

    # 按验证损失升序排序，取前3个作为新的最佳模型
    best_models = sorted(candidates, key=lambda x: x[0])[:3]

    # 提取需要保留的模型路径
    keep_paths = {path for (loss, path) in best_models}

    # 删除不在保留列表中的模型文件
    for fname in os.listdir(MODEL_DIR):
        fpath = os.path.join(MODEL_DIR, fname)
        # 只删除文件，不删除子目录，并且只删除不在保留列表中的模型
        if os.path.isfile(fpath) and fpath.endswith(".pth") and fpath not in keep_paths:
            os.remove(fpath)
            print(f"已删除非最佳模型: {fpath}")

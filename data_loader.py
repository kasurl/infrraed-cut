import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import numpy as np
from PIL import Image

class RoadCrackDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        # 只保留文件名（不含路径），并排序确保一致性
        image_names = sorted(os.listdir(images_dir))
        mask_names = sorted(os.listdir(masks_dir))
        # 校验文件名是否一一对应（可选，增强鲁棒性）
        assert len(image_names) == len(mask_names), "图像和掩码数量不匹配"
        for img_name, mask_name in zip(image_names, mask_names):
            assert img_name.split('.')[0] == mask_name.split('.')[0], f"文件名不匹配: {img_name} 和 {mask_name}"
        # 构建路径
        self.image_filenames = [os.path.join(images_dir, f) for f in image_names]
        self.mask_filenames = [os.path.join(masks_dir, f) for f in mask_names]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_filenames[idx]
        mask_name = self.mask_filenames[idx]
        image = Image.open(img_name).convert("RGB")
        mask = Image.open(mask_name).convert("L")
        # 将掩码转换为二值图像
        mask = np.array(mask)
        # 如果掩码不是二值的，使用阈值进行二值化
        if np.unique(mask).size > 2:  # 如果有多于2个唯一值
            threshold = np.max(mask) * 0.5  # 使用最大值的一半作为阈值
            mask = (mask > threshold).astype(np.uint8) * 255

        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # 确保掩码是二值的 (0或1)
        mask = (mask > 0.5).float()

        # 打印一些调试信息（只在第一次调用时）
        if idx == 0 and not hasattr(self, 'debug_printed'):
            print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
            print(f"Mask shape: {mask.shape}, unique values: {torch.unique(mask)}")
            self.debug_printed = True


        return image, mask

# 示例转换
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])


# 创建数据集实例
train_dataset = RoadCrackDataset(
    images_dir="dataset/roadcrack/train/images/",
    masks_dir="dataset/roadcrack/train/masks/",
    transform=transform
)

val_dataset = RoadCrackDataset(
    images_dir="dataset/roadcrack/val/images/",
    masks_dir="dataset/roadcrack/val/masks/",
    transform=transform
)

test_dataset = RoadCrackDataset(
    images_dir="dataset/roadcrack/test/lwir/",
    masks_dir="dataset/roadcrack/test/masks/",
    transform=transform
)
# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
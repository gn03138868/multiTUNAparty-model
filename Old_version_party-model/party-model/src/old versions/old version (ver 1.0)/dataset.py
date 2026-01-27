import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch

class SegmentationDataset(Dataset):
    def __init__(self, data_root, mode='train', patch_size=400):
        self.image_dir = os.path.join(data_root, mode, 'images')
        self.mask_dir = os.path.join(data_root, mode, 'masks')
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        self.patch_size = patch_size

    def __len__(self):
        return len(self.image_files)

    def _extract_patches(self, image, mask):
        """
        将图像和掩码切割为 400x400 的 patches
        """
        patches = []
        h, w = image.shape[:2]
        
        # 计算切割步长
        stride = self.patch_size
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                # 提取图像 patch
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                # 提取掩码 patch
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                patches.append((img_patch, mask_patch))
        
        return patches

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # 加载图像和掩码
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 确保图像和掩码尺寸一致
        assert image.shape[:2] == mask.shape[:2], \
            f"图像和掩码尺寸不匹配: {image.shape} vs {mask.shape}"
        
        # 提取 patches
        patches = self._extract_patches(image, mask)
        
        # 转换为 Tensor
        patch_tensors = []
        for img_patch, mask_patch in patches:
            # 归一化
            img_patch = img_patch.astype(np.float32) / 255.0
            mask_patch = mask_patch.astype(np.float32) / 255.0
            
            # 转换为 Tensor
            img_tensor = torch.from_numpy(img_patch).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)
            patch_tensors.append((img_tensor, mask_tensor))
        
        return patch_tensors
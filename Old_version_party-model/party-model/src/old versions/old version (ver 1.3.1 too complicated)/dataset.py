# ==== src/dataset.py ====
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁用Albumentations版本檢查
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch

class SegmentationDataset(Dataset):
    def __init__(self, data_root, mode='train', patch_size=128):
        self.image_dir = os.path.join(data_root, mode, 'images')
        self.mask_dir = os.path.join(data_root, mode, 'masks')
        self.patch_size = patch_size
        self.mode = mode
        
        # 定義支持的圖像與mask格式（不分大小寫）
        valid_image_exts = ('.jpg', '.jpeg', '.png')
        valid_mask_exts = ('.jpg', '.jpeg', '.png')
        
        # 加載文件列表並驗證
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(valid_image_exts)])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.lower().endswith(valid_mask_exts)])
        assert len(self.image_files) == len(self.mask_files), "圖像和掩碼數量不匹配"
        
        # 數據增強配置
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ], additional_targets={'mask': 'mask'})
        
        # 預先生成所有patches
        self.patches = []
        self._precompute_patches()

    def _precompute_patches(self):
        for img_name, mask_name in zip(self.image_files, self.mask_files):
            image_path = os.path.join(self.image_dir, img_name)
            mask_path = os.path.join(self.mask_dir, mask_name)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # 驗證尺寸
            assert image.shape[:2] == mask.shape, f"{img_name} 尺寸不匹配"
            
            # 生成patches
            patches = self._extract_patches(image, mask)
            self.patches.extend(patches)

    def _extract_patches(self, image, mask):
        h, w = image.shape[:2]
        stride = self.patch_size
        patches = []
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # 訓練時應用數據增強
                if self.mode == 'train':
                    augmented = self.transform(
                        image=img_patch, 
                        mask=mask_patch
                    )
                    img_patch = augmented['image']
                    mask_patch = augmented['mask']
                
                patches.append((img_patch, mask_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patches[idx]
        
        # 歸一化並轉換為Tensor
        img_tensor = torch.from_numpy(img_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy((mask_patch / 255.0).astype(np.float32)).unsqueeze(0)
        
        return img_tensor, mask_tensor

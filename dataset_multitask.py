"""
多任務分割資料集
支援自動識別或手動指定影像任務類型
"""

import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch
from pathlib import Path

class MultiTaskSegmentationDataset(Dataset):
    """
    多任務分割資料集
    
    資料夾結構方案一（推薦）- 按任務分類：
    data/
    ├── train/
    │   ├── cell/          # 植物細胞
    │   │   ├── images/
    │   │   └── masks/
    │   ├── blood/         # 血球
    │   │   ├── images/
    │   │   └── masks/
    │   └── root/          # 根系
    │       ├── images/
    │       └── masks/
    
    資料夾結構方案二 - 傳統結構（需要在檔名中標記任務）：
    data/
    ├── train/
    │   ├── images/
    │   │   ├── cell_001.jpg
    │   │   ├── blood_001.jpg
    │   │   └── root_001.jpg
    │   └── masks/
    │       ├── cell_001.png
    │       ├── blood_001.png
    │       └── root_001.png
    """
    
    TASK_MAPPING = {
        'cell': 0,    # 植物細胞
        'blood': 1,   # 血球
        'root': 2     # 根系
    }
    
    def __init__(
        self, 
        data_root, 
        mode='train', 
        patch_size=400,
        task_structure='subfolder',  # 'subfolder' 或 'filename'
        augment_params=None
    ):
        """
        Args:
            data_root: 資料根目錄
            mode: 'train', 'val', 或 'test'
            patch_size: patch 大小
            task_structure: 
                - 'subfolder': 任務按子資料夾分類
                - 'filename': 任務標記在檔名中 (e.g., cell_001.jpg)
            augment_params: 自定義的數據增強參數字典
        """
        self.data_root = data_root
        self.mode = mode
        self.patch_size = patch_size
        self.task_structure = task_structure
        
        # 載入所有影像和對應的任務標籤
        self.samples = []  # [(image_path, mask_path, task_id), ...]
        self._load_samples()
        
        # 配置數據增強
        self.augment_configs = self._get_augmentation_configs(augment_params)
        
        # 預先生成所有 patches
        self.patches = []
        self.patch_image_ids = []  # 記錄每個patch來自哪張圖片
        self._precompute_patches()
        
        print(f"Loaded {len(self.patches)} patches from {len(self.samples)} images")
        self._print_task_distribution()
    
    def _load_samples(self):
        """載入所有影像樣本和對應的任務標籤"""
        
        if self.task_structure == 'subfolder':
            # 方案一：按子資料夾分類
            base_path = Path(self.data_root) / self.mode
            
            for task_name, task_id in self.TASK_MAPPING.items():
                task_path = base_path / task_name
                
                if not task_path.exists():
                    continue
                
                image_dir = task_path / 'images'
                mask_dir = task_path / 'masks'
                
                if not image_dir.exists() or not mask_dir.exists():
                    continue
                
                image_files = sorted([f for f in os.listdir(image_dir) 
                                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                
                for img_file in image_files:
                    # 尋找對應的 mask（可能是 .png 或其他格式）
                    mask_file = self._find_matching_mask(img_file, mask_dir)
                    
                    if mask_file:
                        img_path = image_dir / img_file
                        mask_path = mask_dir / mask_file
                        self.samples.append((str(img_path), str(mask_path), task_id))
        
        else:  # 'filename'
            # 方案二：從檔名中識別任務類型
            image_dir = Path(self.data_root) / self.mode / 'images'
            mask_dir = Path(self.data_root) / self.mode / 'masks'
            
            if not image_dir.exists():
                raise ValueError(f"Image directory not found: {image_dir}")
            
            image_files = sorted([f for f in os.listdir(image_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            
            for img_file in image_files:
                # 從檔名中提取任務類型
                task_id = self._extract_task_from_filename(img_file)
                
                mask_file = self._find_matching_mask(img_file, mask_dir)
                
                if mask_file:
                    img_path = image_dir / img_file
                    mask_path = mask_dir / mask_file
                    self.samples.append((str(img_path), str(mask_path), task_id))
    
    def _find_matching_mask(self, img_file, mask_dir):
        """尋找對應的 mask 檔案"""
        base_name = os.path.splitext(img_file)[0]
        
        # 嘗試不同的擴展名
        for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']:
            mask_file = base_name + ext
            if (mask_dir / mask_file).exists():
                return mask_file
        
        return None
    
    def _extract_task_from_filename(self, filename):
        """從檔名中提取任務類型"""
        filename_lower = filename.lower()
        
        for task_name, task_id in self.TASK_MAPPING.items():
            if filename_lower.startswith(task_name):
                return task_id
        
        # 預設為細胞任務
        print(f"Warning: Cannot determine task type from filename '{filename}', defaulting to 'cell'")
        return 0
    
    def _get_augmentation_configs(self, augment_params=None):
        """
        為不同任務配置不同的數據增強策略
        """
        if augment_params:
            # 使用自定義參數
            return {
                0: self._build_transform(augment_params.get('cell', {})),
                1: self._build_transform(augment_params.get('blood', {})),
                2: self._build_transform(augment_params.get('root', {}))
            }
        
        # 預設配置
        configs = {}
        
        # 植物細胞：需要保持多邊形結構，使用適度的形變
        configs[0] = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2)
        ], additional_targets={'mask': 'mask'})
        
        # 血球：圓形結構，需要較少的形變
        configs[1] = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.ElasticTransform(alpha=0.5, sigma=30, p=0.15),  # 較小的形變
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2)  # 添加噪聲模擬染色不均
        ], additional_targets={'mask': 'mask'})
        
        # 根系：線性結構，主要調整對比度，避免過度形變
        configs[2] = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.4, p=0.5),
            # 根系不使用 ElasticTransform 和 GridDistortion
            A.GaussianBlur(blur_limit=(3, 5), p=0.2)  # 輕微模糊模擬不同焦距
        ], additional_targets={'mask': 'mask'})
        
        return configs
    
    def _build_transform(self, params):
        """根據參數構建數據增強"""
        transforms = []
        
        if params.get('rotate', True):
            transforms.append(A.RandomRotate90(p=0.5))
        if params.get('hflip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if params.get('vflip', True):
            transforms.append(A.VerticalFlip(p=0.5))
        if params.get('brightness', True):
            transforms.append(A.RandomBrightnessContrast(p=0.3))
        
        return A.Compose(transforms, additional_targets={'mask': 'mask'})
    
    def _precompute_patches(self):
        """預先生成所有 patches"""
        for img_path, mask_path, task_id in self.samples:
            try:
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    print(f"Warning: Failed to load {img_path} or {mask_path}")
                    continue
                
                # 驗證尺寸
                if image.shape[:2] != mask.shape:
                    print(f"Warning: Size mismatch in {img_path}")
                    continue
                
                # 提取image_id（檔名不含副檔名）
                image_id = Path(img_path).stem  # 例如 "L_15"
                
                # 生成 patches
                patches = self._extract_patches(image, mask, task_id)
                self.patches.extend(patches)
                
                # 為這張圖片的每個patch記錄image_id
                for _ in range(len(patches)):
                    self.patch_image_ids.append(image_id)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    def _extract_patches(self, image, mask, task_id):
        """提取 patches"""
        h, w = image.shape[:2]
        stride = self.patch_size // 2  # 50% 重疊
        patches = []
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # 訓練時應用任務特定的數據增強
                if self.mode == 'train':
                    augmented = self.augment_configs[task_id](
                        image=img_patch, 
                        mask=mask_patch
                    )
                    img_patch = augmented['image']
                    mask_patch = augmented['mask']
                
                patches.append((img_patch, mask_patch, task_id))
        
        return patches
    
    def _print_task_distribution(self):
        """列印任務分佈"""
        task_counts = {0: 0, 1: 0, 2: 0}
        task_images = {0: set(), 1: set(), 2: set()}  # 統計每個任務有多少張獨立圖片
        
        for idx, (_, _, task_id) in enumerate(self.patches):
            task_counts[task_id] += 1
            task_images[task_id].add(self.patch_image_ids[idx])
        
        task_names = {0: 'Cell', 1: 'Blood', 2: 'Root'}
        print(f"\nTask distribution in {self.mode} set:")
        for task_id, count in task_counts.items():
            n_images = len(task_images[task_id])
            print(f"  {task_names[task_id]}: {count} patches from {n_images} images")
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        img_patch, mask_patch, task_id = self.patches[idx]
        image_id = self.patch_image_ids[idx]  # 獲取此patch的image_id
        
        # 歸一化並轉換為 Tensor
        img_tensor = torch.from_numpy(
            img_patch.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        
        mask_tensor = torch.from_numpy(
            (mask_patch / 255.0).astype(np.float32)
        ).unsqueeze(0)
        
        task_tensor = torch.tensor(task_id, dtype=torch.long)
        
        return img_tensor, mask_tensor, task_tensor, image_id  # 返回4個值
    
    @staticmethod
    def get_task_name(task_id):
        """獲取任務名稱"""
        task_names = {0: 'Cell', 1: 'Blood', 2: 'Root'}
        return task_names.get(task_id, 'Unknown')


# ============================================================================
# 測試程式碼
# ============================================================================

if __name__ == '__main__':
    print("Testing MultiTaskSegmentationDataset...")
    
    # 測試資料夾結構方案一
    try:
        dataset = MultiTaskSegmentationDataset(
            data_root='data',
            mode='train',
            patch_size=400,
            task_structure='subfolder'
        )
        
        print(f"\nTotal patches: {len(dataset)}")
        
        # 測試載入一個樣本
        if len(dataset) > 0:
            img, mask, task, image_id = dataset[0]
            print(f"\nSample 0:")
            print(f"  Image shape: {img.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Task: {MultiTaskSegmentationDataset.get_task_name(task.item())}")
            print(f"  Image ID: {image_id}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease check your data structure:")
        print("data/")
        print("├── train/")
        print("│   ├── cell/")
        print("│   │   ├── images/")
        print("│   │   └── masks/")
        print("│   ├── blood/")
        print("│   │   ├── images/")
        print("│   │   └── masks/")
        print("│   └── root/")
        print("│       ├── images/")
        print("│       └── masks/")

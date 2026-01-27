# ==== src/dataset.py ====
import os
import cv2
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
import torch

class SegmentationDataset(Dataset):
    def __init__(self, data_root, mode='train', patch_size=400):
        self.image_dir = os.path.join(data_root, mode, 'images')
        self.mask_dir = os.path.join(data_root, mode, 'masks')
        self.patch_size = patch_size
        self.mode = mode
        
        # 加載文件列表並驗證
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.mask_files = sorted([f for f in os.listdir(self.mask_dir) if f.endswith('.png')])
        assert len(self.image_files) == len(self.mask_files), "image和mask數量不匹配"
        
        # 數據增強配置，這裡使用了各種小技巧，讓模型學一些麻煩奇怪的形狀
        self.transform = A.Compose([
        
            ##隨機將圖像旋轉90度的倍數（即90、180、270或360度）。
            #這是一個離散的旋轉操作，保證旋轉後的圖像依舊不失真，因為90度旋轉通常不會引入插值問題。
            #對於細胞圖像來說，讓模型在不同角度下都能識別相同的細胞結構，提高模型的旋轉不變性。
            A.RandomRotate90(p=0.5),
            
            ##HorizontalFlip (水平翻轉)-將圖像沿垂直軸對調，相當於左右鏡像。
            #這個操作模擬了左右對稱的變化，有助於模型學習在水平方向上不變的特徵。
            #對於細胞圖像，這有助於減少因細胞分布方向帶來的偏差，讓模型對左右方向的細節均能良好識別。
            A.HorizontalFlip(p=0.5),
            
            ##VerticalFlip (垂直翻轉)類似水平翻轉
            A.VerticalFlip(p=0.5),
            
            ##RandomBrightnessContrast-隨機調整圖像的亮度和對比度。
            #通常會在一個預設範圍內隨機增加或減少亮度，使得圖像看起來更亮或更暗；同時對比度也會隨之變化。
            #這個操作模擬了不同照明條件下的拍攝效果，讓模型在面對不同曝光、亮度或對比度情況時，都能穩定識別細胞的邊緣和結構。
            A.RandomBrightnessContrast(p=0.2),
            
            ##彈性變換 (Elastic Transformation)-將圖片進行非剛性的、局部的隨機變形。它模擬了物體的彈性變形，就像真實世界中柔軟物體可能會發生的扭曲與拉伸。
            #通常會先生成一個隨機的位移場（即每個像素都有一個位移向量），然後利用平滑（例如高斯模糊）過程使位移場更加連續和平滑。
            #這個變換可以用參數如 alpha（控制變形強度）、sigma（控制平滑程度）以及 alpha_affine（控制仿射變換的程度）來調整。
            #這種變換對於細胞圖像來說，能模擬細胞在不同環境下的微小變形，同時保留邊緣結構，讓模型學習更靈活的表達。
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),#alpha_affine=50移除，太新了
            
            ##網格扭曲 (Grid Distortion)-將圖像分割成一個格子網，每個格子內的像素通過縮放或扭曲變換，從而在局部產生變形。
            #將圖像劃分成固定數目的小格子（由參數如 num_steps 決定），然後在每個格子的邊緣進行隨機擾動。
            #通過 distort_limit 等參數設置每個格子可以變形的幅度，確保變形幅度適中，不會破壞全局結構。
            #此方法主要用於生成局部微調的變化，幫助模型在遇到真實圖像中出現的局部形狀變化時，依然能準確分辨細節，例如細胞間隙的位置和連續性。
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.2)
        ], additional_targets={'mask': 'mask'})
        
        # 預先生成所有 patches
        self.patches = []
        self._precompute_patches()

    def _precompute_patches(self):
        for img_name, mask_name in zip(self.image_files, self.mask_files):
            image = cv2.cvtColor(cv2.imread(os.path.join(self.image_dir, img_name)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.mask_dir, mask_name), cv2.IMREAD_GRAYSCALE)
            
            # 驗證尺寸
            assert image.shape[:2] == mask.shape, f"{img_name} 尺寸不匹配"
            
            # 生成patches（此處 stride 改為 patch_size//2 ）
            patches = self._extract_patches(image, mask)
            self.patches.extend(patches)

    def _extract_patches(self, image, mask):
        h, w = image.shape[:2]
        stride = self.patch_size // 2   # 使用重疊 patch
        patches = []
        
        for y in range(0, h - self.patch_size + 1, stride):
            for x in range(0, w - self.patch_size + 1, stride):
                img_patch = image[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                
                # 訓練時應用數據增強
                if self.mode == 'train':
                    augmented = self.transform(image=img_patch, mask=mask_patch)
                    img_patch = augmented['image']
                    mask_patch = augmented['mask']
                
                patches.append((img_patch, mask_patch))
        return patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_patch, mask_patch = self.patches[idx]
        
        # 歸一化並轉換為 Tensor
        img_tensor = torch.from_numpy(img_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        mask_tensor = torch.from_numpy((mask_patch / 255.0).astype(np.float32)).unsqueeze(0)
        
        return img_tensor, mask_tensor

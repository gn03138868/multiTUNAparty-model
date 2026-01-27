# ==== src/model.py ====
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class TransUNet(nn.Module):
    def __init__(self, input_size=128, model_patch_size=16):
        """
        :param input_size: 輸入圖像尺寸（例如 128，對應 dataset 的 patch_size）
        :param model_patch_size: ViT 分割小 patch 的尺寸（例如 16）
        """
        super().__init__()
        
        # 更新 ViT 配置：使用更深的 transformer 層數和更高維度的特徵表示
        config = ViTConfig(
            image_size=input_size,
            patch_size=model_patch_size,
            num_channels=3,
            hidden_size=2048,         # 提升特徵維度
            num_hidden_layers=12,       # transformer 層數增加到 12
            num_attention_heads=16,     # 注意力頭數
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        self.vit = ViTModel(config)
        
        # 假設輸入 image_size 為 128，patch_size 為 16，則 ViT 將輸出 [batch, 1+64, hidden_size]
        # 排除 class token 後特徵圖尺寸為 8x8
        
        # 改進的 decoder 結構：上採樣 4 次 (8 -> 16 -> 32 -> 64 -> 128)，並增加卷積與 BatchNorm 層以提升細節恢復
        self.decoder = nn.Sequential(
            # Block 1: 8x8 -> 16x16
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Block 2: 16x16 -> 32x32
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Block 3: 32x32 -> 64x64
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # Block 4: 64x64 -> 128x128
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            # 額外細化層
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)  # 輸出單通道 mask
        )

    def forward(self, x):
        vit_output = self.vit(x)
        features = vit_output.last_hidden_state  # shape: [batch, 1+num_patches, hidden_size]
        
        # 排除 class token 並重塑為 2D 特徵圖
        batch_size, seq_len, hidden_dim = features.shape
        # 假設 patch 數量為正方形，排除第一個 class token
        h = w = int((seq_len - 1) ** 0.5)
        features = features[:, 1:].permute(0, 2, 1).view(batch_size, hidden_dim, h, w)
        
        return self.decoder(features)


